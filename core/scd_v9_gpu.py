import numpy as np
from time import time
from torch.multiprocessing import Pool

import torch

class SCD(object):
    """
    Stochastic coordinate descent for 01 loss optimization
    Numpy version Proto Type
    """

    def __init__(self, nrows, nfeatures, w_inc=100, tol=0.001, local_iter=10,
                 num_iters=100, interval=20, round=100, updated_features=10,
                 adaptive_inc=False, n_jobs=4, num_gpus=0, adv_train=False,
                 eps=1.0, verbose=True):
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
        self.verbose = verbose
        self.w_inc = w_inc  #
        self.tol = tol  #
        self.num_iters = num_iters
        self.local_iter = local_iter
        self.round = round
        self.adv_train = adv_train
        self.eps = eps
        self.w = []
        self.b = []
        self.best_w = None
        self.best_b = None
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
        self.inc_scale = w_inc
        if self.adjust_inc:
            self.step = torch.from_numpy(np.linspace(-2, 2, 10))
        else:
            self.step = torch.Tensor([1, -1])
        self.w_inc = None
        self.ref_full_index = None
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
            # rows = labels.shape[0]
            # index = np.random.permutation(rows)
            # val_size = rows // 5
            # val_data = data[index[:val_size]]
            # val_labels = labels[index[:val_size]]
            # train_data = data[index[val_size:]]
            # train_labels = labels[index[val_size:]]
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
        if self.adv_train:
            rows_sum = rows_sum * 2
            # plus = plus * 2
            # minus = minus * 2

        self.yp = torch.ones((rows_sum, rows_sum), dtype=torch.int8).triu_(0)
        self.ref_full_index = torch.repeat_interleave(
            torch.arange(
                self.updated_features * self.step.shape[0]
            ).view((-1, 1)), rows_sum, dim=1)

        # multi-process
        pool = Pool(self.n_jobs)
        results = []
        for r in range(self.round):
            if warm_start and self.w_index != []:
                column_indices = self.w_index[r]
                w = self.w[:, r]
            else:
                column_indices = np.random.choice(np.arange(orig_cols),
                                                  num_cols, replace=False)
                self.w_index.append(column_indices)
                w = np.random.uniform(-1, 1, size=(num_cols, )).astype(
                    np.float32)
            results.append(
                pool.apply_async(
                    self.single_run_adv if self.adv_train else self.single_run,
                                args=(train_data,
                                      train_labels, plus, minus,
                                      val_data, val_labels, w,
                                      column_indices, r % self.num_gpus)))

        pool.close()
        pool.join()

        for i, result in enumerate(results):
            temp_w, temp_b, temp_obj = result.get()
            if warm_start:
                self.w[:, i] = temp_w
                self.b[:, i] = temp_b
                self.obj[i] = temp_obj
            else:
                self.w.append(temp_w.view((-1, 1)))
                self.b.append(temp_b.view((1, 1)))
                self.obj.append(temp_obj)

        if warm_start is False:
            self.w = torch.cat(self.w, dim=1)
            self.b = torch.cat(self.b, dim=1)
            self.obj = torch.Tensor(self.obj)
        best_index = self.obj.argmax()
        self.best_acc = self.obj[best_index]
        self.best_w = self.w[:, best_index]
        self.best_b = self.b[:, best_index]
        self.best_w_index = self.w_index[best_index]

    def single_run(self, data, labels, plus, minus, val_data, val_labels, w,
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
        if type(w) is not torch.Tensor:
            w = torch.from_numpy(w).cuda(self.device)
        else:
            w = w.cuda(self.device)
        # L2 normalization
        temp_w = w / w.norm()

        # self.w_inc = self.inc_scale * self.step
        for i in range(self.num_iters):
            self.w_inc = self.inc_scale * self.step
            # pick rows randomly
            row_index = np.hstack([
                np.random.choice(self.plus_row_index, plus, replace=False),
                np.random.choice(self.minus_row_index, minus, replace=False)
            ])

            temp_w, temp_b = self.single_iteration(
                temp_w, data[row_index].cuda(self.device),
                labels[row_index].cuda(self.device), plus, minus)

            temp_acc = self.eval(data.cuda(self.device),
                                 labels.cuda(self.device),
                                 temp_w, temp_b, batch_size=None)


            # print('%d iterations, test acc: %.5f' %(i, test_acc))
            if temp_acc > best_acc:
                w = temp_w.clone()
                b = temp_b.clone()
                best_acc = temp_acc
                # if self.adjust_inc:
                #     self.w_inc = self.step * w.std()
                test_acc = self.eval(val_data.cuda(self.device),
                                     val_labels.cuda(self.device),
                                     temp_w, temp_b, batch_size=None)
                if self.verbose:
                    print('%d iterations, best acc: %.5f, test acc: %.5f'
                          % (i, best_acc, test_acc))
        del data, val_data
        return w.cpu(), b.cpu(), best_acc

    def single_run_adv(self, data, labels, plus, minus, val_data, val_labels, w,
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
        data = data[:, column_indices].cuda(self.device)
        val_data = val_data[:, column_indices]

        # w = np.random.normal(0, 1, size=(data.shape[1],)).astype(np.float32)
        if type(w) is not torch.Tensor:
            w = torch.from_numpy(w).cuda(self.device)
        else:
            w = w.cuda(self.device)
        # L2 normalization
        temp_w = w / w.norm()
        temp_b = 0
        # self.w_inc = self.inc_scale * self.step
        for i in range(self.num_iters):
            data_adv = self.get_adv(data, temp_w, temp_b)
            self.w_inc = self.inc_scale * self.step
            # pick rows randomly
            row_index = np.hstack([
                np.random.choice(self.plus_row_index, plus, replace=False),
                np.random.choice(self.minus_row_index, minus, replace=False)
            ])
            train_data = torch.cat([data[row_index], data_adv[row_index]], dim=0)
            train_labels = torch.cat([labels[row_index], labels[row_index]], dim=0)

            temp_w, temp_b = self.single_iteration(
                temp_w, train_data, train_labels.cuda(self.device), plus, minus)

            temp_acc = self.eval(torch.cat([data, data_adv]),
                                 torch.cat([labels, labels], dim=0).cuda(self.device),
                                 temp_w, temp_b, batch_size=None)


            # print('%d iterations, test acc: %.5f' %(i, test_acc))
            if temp_acc > best_acc:
                w = temp_w.clone()
                b = temp_b.clone()
                best_acc = temp_acc
                # if self.adjust_inc:
                #     self.w_inc = self.step * w.std()
                test_acc = self.eval(val_data.cuda(self.device),
                                     val_labels.cuda(self.device),
                                     temp_w, temp_b, batch_size=None)
                print('%d iterations, best acc: %.5f, test acc: %.5f'
                      % (i, best_acc, test_acc))
        del data, val_data
        return w.cpu(), b.cpu(), best_acc

    def get_adv(self, data, w, b):

        train_data_adv = torch.zeros_like(data)
        yp = (torch.sign(data.matmul(w) + b) + 1) // 2
        train_data_adv += self.eps * (yp.view((-1, 1)) * 2 - 1) * w.reshape((1, -1))
        train_data_adv += data
        torch.clamp(train_data_adv, 0, 1, out=train_data_adv)

        return train_data_adv

    def single_iteration(self, w, data, labels, plus, minus):
        """

        :param data: subset of training data
        :param labels: subset of training labels
        :param plus: number of +1 data points
        :param minus: number of -1 data points
        :return: temporary w and b we find in this subset
        """

        temp_best_objective = 0
        best_w, best_b = self.get_best_w_and_b(data, labels, w, plus, minus,
                                               temp_best_objective)

        return best_w, best_b

    def argsort(self, x, dim=-1):

        return torch.argsort(x, dim=dim)

    def get_best_w_and_b(self, data, labels, w, plus, minus, best_objective):
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

        projection = self.obtain_projection(data, w)
        raw_index_sorted = self.argsort(projection)
        projection = projection[raw_index_sorted]
        best_acc, b, best_index = self.get_best_b(
            labels[raw_index_sorted], projection, None, None, plus, minus)

        localit = 0

        # updation_order = np.random.choice(
        #     np.arange(w.shape[0]), self.updated_features, False
        # )

        while localit < self.local_iter and \
                    best_acc - best_objective > self.tol:
            # print('inner loop while. %d localit' % localit)

            updation_order = np.random.choice(
                np.arange(w.shape[0]), self.updated_features, False
            )
            best_objective = best_acc
            inc = []
            best_b = b
            best_temp_index = best_index

            for i in range(self.w_inc.shape[0]):
                w_inc = self.w_inc[i]
                w_ = torch.repeat_interleave(
                    w.reshape((-1, 1)), self.updated_features, dim=1)
                w_[updation_order, np.arange(self.updated_features)] += w_inc
                inc.append(w_)
            w_ = torch.cat(inc, dim=1)
            del inc
            w_ /= w_.norm(dim=0)
            projection = self.obtain_projection(data, w_).T  # Transpose
            raw_index_sorted = self.argsort(projection, dim=1)
            projection = projection[self.ref_full_index, raw_index_sorted]
            temp_labels = labels[raw_index_sorted]

            temp_acc, temp_b, temp_row, temp_index = self.get_best_b(
                temp_labels, projection, best_temp_index,
                self.interval, plus, minus, group=True
            )

            if temp_acc > best_acc:
                # print('update..0')
                best_acc = temp_acc
                best_w_inc = self.w_inc[temp_row//self.updated_features]
                if self.adjust_inc:
                    self.w_inc = best_w_inc * self.step
                # print(best_w_inc)
                best_b = temp_b
                best_temp_index = temp_index
                w = w_[:, temp_row]
            # delete variables
            del w_, projection, raw_index_sorted, temp_labels
            localit += 1
        # print('while loop %d times, best acc: %.5f' % (localit, best_acc))
        return w, best_b

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

        gt = torch.unsqueeze(labels, dim=1)
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

    def get_best_b(self, labels, projection, index, interval, plus, minus,
                   group=False):
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
                yp = torch.unsqueeze(self.yp[start_index: end_index],
                                     dim=0).cuda(self.device)
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

    def eval(self, data, labels, w, b, batch_size):
        """

        :param data:
        :param labels:
        :param w_matrix:
        :param b_matrix:
        :param batch_size:
        :return:
        """
        if batch_size is None:
            yp = (torch.sign(data.matmul(w) + b) + 1) // 2
            acc = (yp == labels).float().mean()

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
            yp = (torch.sign(x[:, self.best_w_index].matmul(self.best_w) +
                          self.best_b) + 1) // 2

        elif kind == 'vote':
            yp = torch.zeros((x.shape[0], self.round), dtype=torch.float)
            for i in range(self.round):
                yp[:, i] = (torch.sign(x[:, self.w_index[i]].matmul(
                    self.w[:, i]) + self.b[:, i]) + 1) // 2

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

    np.random.seed(20156)
    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int8)
    train_data, test_data, train_label, test_label = train_test_split(x, y)

    scd = SCD(nrows=0.2, nfeatures=1, w_inc=1, tol=0.000001, local_iter=100,
              num_iters=100, round=2, interval=10, adaptive_inc=False,
              updated_features=10, n_jobs=2, num_gpus=1)
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