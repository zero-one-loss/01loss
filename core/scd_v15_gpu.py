import numpy as np
from time import time
from torch.multiprocessing import Pool
import torch


class SCD(object):
    """
    Stochastic coordinate descent for 01 loss optimization
    Numpy version Proto Type
    """

    def __init__(self, nrows, nfeatures, w_inc1=0.1, w_inc2=0.1, tol=0.001,
                 local_iter=10, verbose=True,
                 num_iters=100, interval=20, round=100, updated_features=10,
                 adaptive_inc=False, n_jobs=4, num_gpus=0, hidden_nodes=10):
        """

        :param nrows: ratio of training data in each iteration
        :param nfeatures: ratio of features in each iteration
        :param w_inc: w increment
        :param tol: stop threshold
        :param local_iter: the maximum number of iterations of updating all
                            columns
        :param num_iters: the number of iterations in each RR
        :param interval: interval in bias search if best index given
        :param round: number of round, RR
        :param seed: random seed
        :param n_jobs: number of process
        """
        self.nrows = nrows  #
        self.nfeatures = nfeatures  #
        # self.w_inc1 = w_inc1  #
        # self.w_inc2 = w_inc2
        self.verbose = verbose
        self.tol = tol  #
        self.num_iters = num_iters
        self.local_iter = local_iter
        self.hidden_nodes = hidden_nodes
        self.round = round
        self.w1 = []
        self.b1 = []
        self.w2 = []
        self.b2 = []
        self.best_w1 = None
        self.best_b1 = None
        self.best_w2 = None
        self.best_b2 = None
        self.best_acc = None
        self.best_w_index = None
        self.w_index_order = None
        self.obj = []
        self.orig_plus = 0
        self.orig_minus = 0
        self.plus_row_index = []
        self.minus_row_index = []
        self.yp = None
        # self.warm_start = warm_start
        self.interval = interval
        self.adjust_inc = adaptive_inc
        self.inc_scale1 = w_inc1
        self.inc_scale2 = w_inc2
        if self.adjust_inc:
            self.step = torch.from_numpy(np.linspace(-2, 2, 10))
        else:
            self.step = torch.Tensor([1, -1])
        self.w_inc1 = None
        self.w_inc2 = None
        self.ref_full_index1 = None
        self.ref_full_index2 = None
        self.w_index = []
        self.n_jobs = n_jobs
        self.w_inc_stats = []
        self.updated_features = updated_features
        self.device = None
        self.num_gpus = num_gpus

    def train(self, data, labels, val_data=None, val_labels=None,
              warm_start=False):
        """

        :param data:
        :param labels:
        :param val_data:
        :param val_labels:
        :param warm_start:
        :return:
        """
        # initialize variable
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).char()
        if val_data is None:
            train_data, val_data = data, data
            train_labels, val_labels = labels, labels
        else:
            train_data = data
            train_labels = labels
            val_data = torch.from_numpy(val_data).float()
            val_labels = torch.from_numpy(val_labels).char()

        orig_cols = train_data.size(1)

        # counting class and get their index
        self.plus_row_index = []
        self.minus_row_index = []
        self.orig_minus = 0
        self.orig_plus = 0
        for idx, value in enumerate(train_labels):
            if value == 1:
                self.orig_plus += 1
                self.plus_row_index.append(idx)
            else:
                self.orig_minus += 1
                self.minus_row_index.append(idx)

        # balanced pick rows and cols
        plus = max(2, int(self.orig_plus * self.nrows))
        minus = max(2, int(self.orig_minus * self.nrows))
        num_cols = max(min(5, orig_cols), int(self.nfeatures * orig_cols))

        # initialize up triangle matrix and reference index
        rows_sum = plus + minus
        self.yp = torch.ones((rows_sum, rows_sum), dtype=torch.int8).triu_(0)
        self.ref_full_index1 = torch.repeat_interleave(
            torch.arange(
                self.updated_features * self.step.shape[0]
            ).view((-1, 1)), rows_sum, dim=1)
        self.ref_full_index2 = torch.repeat_interleave(
            torch.arange(
                self.hidden_nodes * self.step.shape[0]
            ).view((-1, 1)), rows_sum, dim=1)
        # multi-process
        pool = Pool(self.n_jobs)
        results = []
        for r in range(self.round):
            if warm_start and self.w_index != []:
                column_indices = self.w_index[r]
                w1 = self.w1[:, :, r]
                w2 = self.w2[:, r]
            else:
                column_indices = np.random.choice(np.arange(orig_cols),
                                                  num_cols, replace=False)
                self.w_index.append(column_indices)
                w1 = np.random.uniform(-1, 1, size=(
                num_cols, self.hidden_nodes)).astype(np.float32)
                w2 = np.random.uniform(-1, 1, size=(self.hidden_nodes,)).astype(
                    np.float32)
            results.append(
                pool.apply_async(self.single_run, args=(
                train_data, train_labels, plus, minus, val_data, val_labels, w1,
                w2, column_indices, r % self.num_gpus)))

        pool.close()
        pool.join()

        for i, result in enumerate(results):
            temp_w1, temp_b1, temp_w2, temp_b2, temp_obj = result.get()
            #            temp_w1, temp_b1, temp_w2, temp_b2, temp_obj = self.single_run(train_data, train_labels, plus, minus, val_data, val_labels, w1, w2, column_indices, r % self.num_gpus)
            if warm_start:
                self.w1[:, :, i] = temp_w1
                self.w2[:, i] = temp_w2
                self.b1[:, i] = temp_b1
                self.b2[i] = temp_b2
                self.obj[i] = temp_obj
            else:
                self.w1.append(temp_w1)
                self.w2.append(temp_w2)
                self.b1.append(temp_b1)
                self.b2.append(temp_b2)
                self.obj.append(temp_obj)

        if warm_start is False:
            self.w1 = torch.stack(self.w1, dim=2)
            self.w2 = torch.stack(self.w2, dim=1)
            self.b1 = torch.stack(self.b1, dim=1)
            self.b2 = torch.Tensor(self.b2)
            self.obj = torch.Tensor(self.obj)
        best_index = self.obj.argmax()
        self.best_acc = self.obj[best_index]
        self.best_w1 = self.w1[:, :, best_index]
        self.best_w2 = self.w2[:, best_index]
        self.best_b1 = self.b1[:, best_index]
        self.best_b2 = self.b2[best_index]
        self.best_w_index = self.w_index[best_index]

        return

    def single_run(self, data, labels, plus, minus, val_data, val_labels, w1,
                   w2,
                   column_indices, device):
        """

        :param data: subset of training data
        :param labels: subset of training labels
        :param plus: number of +1 data points
        :param minus: number of -1 data points
        :return: best w, b, and acc we searched in this subset and evaluate on
                the full training set
        """
        best_acc = 0
        self.device = device
        data = data[:, column_indices]
        val_data = val_data[:, column_indices]
        # w = np.random.normal(0, 1, size=(data.shape[1],)).astype(np.float32)
        if type(w1) is not torch.Tensor:
            w1 = torch.from_numpy(w1).cuda(self.device)
        if type(w2) is not torch.Tensor:
            w2 = torch.from_numpy(w2).cuda(self.device)
        else:
            w1 = w1.cuda(self.device)
            # L2 normalization
            w2 = w2.cuda(self.device)
            # L2 normalization
        temp_w1 = w1 / w1.norm(dim=0)
        temp_w2 = w2 / w2.norm()
        temp_b1 = None
        # self.w_inc = self.inc_scale * self.step
        for i in range(self.num_iters):
            self.w_inc1 = self.inc_scale1 * self.step
            self.w_inc2 = self.inc_scale2 * self.step
            # pick rows randomly
            row_index = np.hstack([
                np.random.choice(self.plus_row_index, plus, replace=False),
                np.random.choice(self.minus_row_index, minus, replace=False)
            ])
            temp_w1, temp_b1, temp_w2, temp_b2 = self.single_iteration(
                temp_w1, temp_w2, None, data[row_index].cuda(self.device),
                labels[row_index].cuda(self.device),
                plus, minus)

            temp_acc = self.eval(data.cuda(self.device),
                                 labels.cuda(self.device),
                                 temp_w1, temp_b1, temp_w2, temp_b2,
                                 batch_size=None)

            #            print('%d iterations, temp acc: %.5f' %(i, temp_acc))
            if temp_acc > best_acc:
                best_w1 = temp_w1.clone()
                best_b1 = temp_b1.clone()
                best_w2 = temp_w2.clone()
                best_b2 = temp_b2.clone()
                best_acc = temp_acc
                # if self.adjust_inc:
                #                     self.w_inc = self.step * w.std()
                test_acc = self.eval(val_data.cuda(self.device),
                                     val_labels.cuda(self.device),
                                     temp_w1, temp_b1, temp_w2, temp_b2,
                                     batch_size=None)
                if self.verbose:
                    print('%d iterations, best acc: %.5f, test acc: %.5f'
                          % (i, temp_acc, test_acc))

        del data, val_data, temp_w1, temp_w2, temp_b1, temp_b2, temp_acc, row_index

        return best_w1.cpu(), best_b1.cpu(), best_w2.cpu(), best_b2.cpu(), best_acc

    def single_iteration(self, w1, w2, b1, data, labels, plus, minus):
        """

        :param data: subset of training data
        :param labels: subset of training labels
        :param plus: number of +1 data points
        :param minus: number of -1 data points
        :return: temporary w and b we find in this subset
        """

        obj = 0
        if b1 is None:
            b1, b2, best_acc = self.get_init_bias(data, labels, w1, w2, plus, minus)
        w2, b2 = self.get_best_w2_and_b2(data, labels, w1, w2, b1, plus, minus,
                                         obj)
        w1, b1 = self.get_best_w1_and_b1(data, labels, w1, w2, b1, b2, plus,
                                         minus, obj)

        return w1, b1, w2, b2

    def argsort(self, x, dim=-1):

        return torch.argsort(x, dim=dim)

    def get_best_w1_and_b1(self, data, labels, w1, w2, b1, b2, plus, minus,
                           best_objective):
        """

        :param iter: ignore
        :param data:
        :param labels:
        :param w:
        :param plus:
        :param minus:
        :param best_objective:
        :return:
        """

        # idx = np.random.choice(np.arange(self.hidden_nodes), 1)[0]
        for idx in np.random.permutation(self.hidden_nodes)[0:1]:
            b1[idx] = 0
            temp_w2 = w2.clone()
            temp_w2[idx] = 0
            projection1 = self.obtain_projection(data, w1)
            p1 = projection1 + b1.view((1, -1))
            pi = projection1[:, idx]
            raw_index_sorted = self.argsort(pi)
            # p1 = p1[raw_index_sorted]
            p2_sign = ((self.obtain_projection(
                (p1[raw_index_sorted].sign() + 1) / 2, temp_w2).view(
                (1, -1)) + b2 + self.yp.cuda(self.device) * w2[
                            idx]).sign() + 1) // 2
            best_acc, b1_i, best_index = self.get_best_b1(labels[raw_index_sorted],
                                                          pi[raw_index_sorted],
                                                          None, None, plus, minus,
                                                          False, p2_sign)
            b1[idx] = b1_i
            del p2_sign, pi
            localit = 0

            # updation_order = np.random.choice(
            #     np.arange(w.shape[0]), self.updated_features, False
            # )

            while localit < self.local_iter and \
                    best_acc - best_objective > self.tol:
                #            print('w1 b1 inner loop while. %d localit' % localit)

                updation_order = np.random.choice(
                    np.arange(w1.shape[0]), self.updated_features, False
                )
                best_objective = best_acc
                inc = []
                best_b = b1
                best_temp_index = best_index

                for i in range(self.w_inc1.shape[0]):
                    w_inc = self.w_inc1[i]
                    w_ = torch.repeat_interleave(
                        w1[:, idx].reshape((-1, 1)), self.updated_features, dim=1)
                    w_[updation_order, np.arange(self.updated_features)] += w_inc
                    inc.append(w_)
                w_ = torch.cat(inc, dim=1)
                del inc, w_inc
                w_ /= w_.norm(dim=0)
                projection = self.obtain_projection(data, w_).T  # Transpose
                raw_index_sorted = self.argsort(projection, dim=1)
                projection = projection[self.ref_full_index1, raw_index_sorted]
                temp_labels = labels[raw_index_sorted]
                p1_sign = (p1.sign() + 1) / 2
                p1_sign[:, idx] = 0
                p2_sign = self.obtain_projection(p1_sign, temp_w2) + b2
                start_index = max(0, best_temp_index - self.interval)
                end_index = min(temp_labels.shape[1],
                                best_temp_index + self.interval)
                p2_sign = ((p2_sign[raw_index_sorted].unsqueeze(dim=1) +
                            self.yp[start_index: end_index].unsqueeze(
                    dim=0).cuda(self.device) * w2[idx]).sign() + 1) // 2
                temp_acc, temp_b, temp_row, temp_index = self.get_best_b1(
                    temp_labels, projection, best_temp_index, self.interval, plus,
                    minus, True, p2_sign)

                if temp_acc > best_acc:
                    # print('update..0')
                    best_acc = temp_acc
                    best_w_inc = self.w_inc1[temp_row // self.updated_features]
                    if self.adjust_inc:
                        self.w_inc = best_w_inc * self.step
                    # print(best_w_inc)
                    b1[idx] = temp_b
                    best_temp_index = temp_index
                    w1[:, idx] = w_[:, temp_row]
                # delete variables
                del w_, projection, raw_index_sorted, temp_labels
                localit += 1
            # print('while loop %d times, best acc: %.5f' % (localit, best_acc))
        return w1, b1

    def get_init_bias(self, data, labels, w1, w2, plus, minus):
        """

        :param data:
        :param labels:
        :param w1:
        :param w2:
        :param plus:
        :param minus:
        :return:
        """

        # best_b1_index_list = [None] * self.hidden_nodes
        projection1 = self.obtain_projection(data, w1)
        init_b1 = projection1.mean(dim=0)
        projection2 = self.obtain_projection(
            ((projection1 + init_b1.view((1, -1))).sign() + 1) / 2, w2)
        raw_index_sorted = self.argsort(projection2)
        projection2 = projection2[raw_index_sorted]
        best_acc, b2, best_b2_index = self.get_best_b2(labels[raw_index_sorted],
                                                       projection2, None, None,
                                                       plus, minus)
        del projection2, raw_index_sorted

        for i in np.random.permutation(self.hidden_nodes):
            init_b1[i] = 0
            temp_w2 = w2.clone()
            temp_w2[i] = 0
            p1 = projection1 + init_b1.view((1, -1))
            pi = projection1[:, i]
            raw_index_sorted = self.argsort(pi)
            p1 = p1[raw_index_sorted]
            p2_sign = ((self.obtain_projection((p1.sign() + 1) / 2,
                                               temp_w2).view(
                (1, -1)) + b2 + self.yp.cuda(self.device) * w2[
                            i]).sign() + 1) // 2
            best_acc, b1_i, best_b1_i_index = self.get_best_b1(
                labels[raw_index_sorted], pi[raw_index_sorted], None, None,
                plus, minus, False, p2_sign)
            init_b1[i] = b1_i
            # best_b1_index_list[i] = best_b1_i_index
            del p1, pi, p2_sign, raw_index_sorted, temp_w2

        return init_b1, b2, best_acc

    def get_best_b1(self, labels, projection, index, interval, plus, minus,
                    group=False, yp=None):
        if group:
            gt = labels.clone()
            if index is None:
                yp = torch.unsqueeze(self.yp, dim=0).cuda(self.device)
                # return best acc coordinate
                best_index_coord, acc = self.cal_acc_group(gt, yp, plus, minus)
                row, best_index = best_index_coord[0], best_index_coord[1]
                if best_index_coord[1] == 0:
                    b = -1 * projection[row][best_index] + 0.01
                else:
                    b = -1 * projection[row][best_index - 1: best_index].mean()
            else:
                start_index = max(0, index - interval)
                # end_index = min(gt.shape[1], index + interval)
                # if yp is not None:
                #     yp = yp[:, start_index: end_index]
                # return best acc coordinate
                best_index_coord, acc = self.cal_acc_group(gt, yp, plus, minus)
                row, best_index = best_index_coord[0], best_index_coord[1]
                best_index += start_index
                if best_index_coord[1] == 0:
                    b = -1 * projection[row][best_index] + 0.01
                else:
                    b = -1 * projection[row][best_index - 1: best_index].mean()

            return acc, b, row, best_index

        else:
            gt = labels.clone()
            if index is None:
                if yp is None:
                    yp = self.yp.cuda(self.device)
                best_index, acc = self.cal_acc(gt, yp, plus, minus)
                if best_index == 0:
                    b = -1 * projection[best_index] + 0.01
                else:
                    b = -1 * projection[best_index - 1: best_index].mean()

            else:
                start_index = max(0, index - interval)
                end_index = min(gt.shape[0], index + interval)
                yp = self.yp[start_index: end_index].cuda(self.device)
                best_index, acc = self.cal_acc(gt, yp, plus, minus)
                best_index += start_index
                if best_index == 0:
                    b = -1 * projection[best_index] + 0.01
                else:
                    b = -1 * projection[best_index - 1: best_index].mean()

            return acc, b, best_index

    def get_best_w2_and_b2(self, data, labels, w1, w2, initial_b1, plus, minus,
                           best_objective):
        """

        :param data:
        :param labels:
        :param w1:
        :param s2:
        :param initial_b2:
        :param best_b2_index:
        :param plus:
        :param minus:
        :param best_objective:
        :return:
        """
        p1_sign = ((self.obtain_projection(data, w1) + initial_b1.view(
            (1, -1))).sign() + 1) / 2
        projection = self.obtain_projection(p1_sign, w2)
        raw_index_sorted = self.argsort(projection)
        projection = projection[raw_index_sorted]
        best_acc, b, best_index = self.get_best_b2(
            labels[raw_index_sorted], projection, None, None, plus, minus)

        localit = 0
        del raw_index_sorted, projection

        # updation_order = np.random.choice(
        #     np.arange(w.shape[0]), self.updated_features, False
        # )
        best_b = b
        while localit < self.local_iter and \
                best_acc - best_objective > self.tol:
            #            print('w2 b2 inner loop while. %d localit' % localit)

            # updation_order = np.random.choice(
            #     np.arange(w2.shape[0]), self.updated_features, False
            # )
            updation_order = np.random.permutation(self.hidden_nodes)
            best_objective = best_acc
            inc = []
            best_b = b
            best_temp_index = best_index

            for i in range(self.w_inc2.shape[0]):
                w_inc = self.w_inc2[i]
                w_ = torch.repeat_interleave(
                    w2.reshape((-1, 1)), self.hidden_nodes, dim=1)
                w_[updation_order, np.arange(self.hidden_nodes)] += w_inc
                inc.append(w_)
            w_ = torch.cat(inc, dim=1)
            del inc, w_inc
            w_ /= w_.norm(dim=0)
            projection = self.obtain_projection(p1_sign, w_).T  # Transpose
            raw_index_sorted = self.argsort(projection, dim=1)
            projection = projection[self.ref_full_index2, raw_index_sorted]
            temp_labels = labels[raw_index_sorted]

            temp_acc, temp_b, temp_row, temp_index = self.get_best_b2(
                temp_labels, projection, best_temp_index,
                self.interval, plus, minus, group=True
            )

            if temp_acc > best_acc:
                # print('update..0')
                best_acc = temp_acc
                best_w_inc = self.w_inc2[temp_row // self.hidden_nodes]
                if self.adjust_inc:
                    self.w_inc2 = best_w_inc * self.step
                # print(best_w_inc)
                best_b = temp_b
                best_temp_index = temp_index
                w2 = w_[:, temp_row]
            # delete variables
            del w_, projection, raw_index_sorted, temp_labels
            localit += 1
        # print('while loop %d times, best acc: %.5f' % (localit, best_acc))
        return w2, best_b

    def cal_acc(self, labels, yp, plus, minus):

        gt = labels.reshape((1, -1))
        sum_ = yp + gt
        plus_correct = (sum_ == 2).sum(dim=1).float()
        minus_correct = (sum_ == 0).sum(dim=1).float()

        # balanced accuracy formula
        acc = (plus_correct / plus + minus_correct / minus) / 2.0
        best_index = acc.argmax().item()

        return best_index, acc[best_index]

    def cal_acc_group(self, labels, yp, plus, minus):

        gt = torch.unsqueeze(labels, dim=1)  # (256 x 1 x 100k)
        sum_ = yp + gt
        plus_correct = (sum_ == 2).sum(dim=2).float()
        minus_correct = (sum_ == 0).sum(dim=2).float()

        # balanced accuracy formula
        acc = (plus_correct / plus + minus_correct / minus) / 2.0
        acc = acc.cpu().numpy()
        best_index = np.unravel_index(np.argmax(acc, axis=None), acc.shape)

        return best_index, acc[best_index]

    def obtain_projection(self, x, w):

        return torch.matmul(x, w)

    def get_best_b2(self, labels, projection, index, interval, plus, minus,
                    group=False, yp=None):
        """

        :param labels:
        :param projection:
        :param index:
        :param raw_index_sorted:
        :param interval:
        :param plus:
        :param minus:
        :return:
        """
        if group:
            gt = labels.clone()
            if index is None:
                yp = torch.unsqueeze(self.yp, dim=0).cuda(self.device)
                # return best acc coordinate
                best_index_coord, acc = self.cal_acc_group(gt, yp, plus, minus)
                row, best_index = best_index_coord[0], best_index_coord[1]
                if best_index_coord[1] == 0:
                    b = -1 * projection[row][best_index] + 0.01
                else:
                    b = -1 * projection[row][best_index - 1: best_index].mean()
            else:
                start_index = max(0, index - interval)
                end_index = min(gt.shape[1], index + interval)
                if yp is None:
                    yp = torch.unsqueeze(
                        self.yp[start_index: end_index].cuda(self.device),
                        dim=0)
                    # (1 x 20 x 100k)
                # return best acc coordinate
                best_index_coord, acc = self.cal_acc_group(gt, yp, plus, minus)
                row, best_index = best_index_coord[0], best_index_coord[1]
                best_index += start_index
                if best_index_coord[1] == 0:
                    b = -1 * projection[row][best_index] + 0.01
                else:
                    b = -1 * projection[row][best_index - 1: best_index].mean()

            return acc, b, row, best_index

        else:
            gt = labels.clone()
            if index is None:
                if yp is None:
                    yp = self.yp.cuda(self.device)
                best_index, acc = self.cal_acc(gt, yp, plus, minus)
                if best_index == 0:
                    b = -1 * projection[best_index] + 0.01
                else:
                    b = -1 * projection[best_index - 1: best_index].mean()

            else:
                start_index = max(0, index - interval)
                end_index = min(gt.shape[0], index + interval)
                yp = self.yp[start_index: end_index].cuda(self.device)
                best_index, acc = self.cal_acc(gt, yp, plus, minus)
                best_index += start_index
                if best_index == 0:
                    b = -1 * projection[best_index] + 0.01
                else:
                    b = -1 * projection[best_index - 1: best_index].mean()

            return acc, b, best_index

    def eval(self, data, labels, w1, b1, w2, b2, batch_size):
        """

        :param data:
        :param labels:
        :param w_matrix:
        :param b_matrix:
        :param batch_size:
        :return:
        """
        if batch_size is None:
            p1 = (torch.sign(data.matmul(w1) + b1.view((1, -1))) + 1) / 2
            yp = (torch.sign(p1.matmul(w2) + b2) + 1) // 2
            acc = (yp == labels).float().mean()
            del p1, yp
            return acc.item()

    def predict(self, x, kind='best', prob=False, all=False):
        """

        :param x:
        :param kind:
        :param prob:
        :return:
        """
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        if kind == 'best':
            yp1 = (torch.sign(x[:, self.best_w_index].matmul(self.best_w1) +
                              self.best_b1.view((1, -1))) + 1) / 2
            yp = (torch.sign(yp1.matmul(self.best_w2) +
                             self.best_b2) + 1) // 2

        elif kind == 'vote':
            yp = torch.zeros((x.shape[0], self.round), dtype=torch.float)
            for i in range(self.round):
                yp1 = (torch.sign(x[:, self.w_index[i]].matmul(
                    self.w1[:, :, i]) + self.b1[:, i].view((1, -1))) + 1) / 2
                yp[:, i] = (torch.sign(yp1.matmul(self.w2[:, i]) +
                                       self.b2[i]) + 1) // 2
            if prob:
                return yp.mean(dim=1)
            if all:
                return yp.cpu().numpy()
            yp = yp.mean(dim=1).round().char()

        return yp.cpu().numpy()

    def predict_best_onehot(self, x):

        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        yp = torch.zeros((x.shape[0], self.round), dtype=torch.float)
        for i in range(self.round):
            yp[:, i] = (torch.sign(x[:, self.w_index[i]].matmul(self.w[:, i]) +
                                   self.b[:, i]) + 1) // 2

        yp = yp.mean(dim=1).round().char()
        target = torch.nn.functional.one_hot(yp.long())

        return target.cpu().numpy()

    def predict_vote_onehot(self, x):

        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        yp = (torch.sign(x[:, self.best_w_index].matmul(self.best_w) +
                         self.best_b) + 1) // 2
        target = torch.nn.functional.one_hot(yp.long())

        return target.cpu().numpy()

    def predict_projection(self, x):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        yp = x[:, self.best_w_index].matmul(self.best_w) + self.best_b

        return yp.cpu().numpy()

    def val(self, x, y):

        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        if type(y) is not torch.Tensor:
            x = torch.from_numpy(y)

        yp = torch.zeros((self.round, x.shape[0])).float()
        for i in range(self.round):
            yp[i] = (torch.sign(x[:, self.w_index[i]].matmul(self.w[:, i]) +
                                self.b[:, i]) + 1) // 2
        # yp = yp.T
        acc = ((yp - y.reshape((1, -1))) == 0).mean(axis=1)

        return acc, acc.max().item()


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    np.random.seed(20126)
    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int8)
    train_data, test_data, train_label, test_label = train_test_split(x, y)

    scd = SCD(nrows=0.8, nfeatures=0.8, w_inc1=1, w_inc2=1, tol=0.0000001,
              local_iter=10,
              num_iters=200, interval=10, round=1, updated_features=10,
              adaptive_inc=False, n_jobs=1, num_gpus=2, hidden_nodes=10)
    a = time()
    scd.train(train_data, train_label, test_data, test_label)

    print('cost %.3f seconds' % (time() - a))
    yp = scd.predict(test_data)

    print('Accuracy: ',
          accuracy_score(y_true=train_label, y_pred=scd.predict(train_data)))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
    yp = scd.predict(test_data, kind='vote')
    print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
    # output probability
    yp = scd.predict(test_data, kind='vote', prob=True)