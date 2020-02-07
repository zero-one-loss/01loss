import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn
import time
import torch.optim as optim
import os
import sys

sys.path.append('..')
from core.scd_v9_gpu import SCD
from blackbox_attack.models import LeNet, CNNModel, LinearModel, Rh, StlModel
import argparse
from sklearn.metrics import accuracy_score
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

parser = argparse.ArgumentParser(description='SCD 01 loss')

parser.add_argument('--epsilon', default=0.3, type=float,
                    help='ratio of rows in each iteration')
parser.add_argument('--Lambda', default=0.1, type=float,
                    help='ratio of features in each vote')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate for substitute model')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device')
parser.add_argument('--target', default='scd', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--random-sign', default=0, type=int,
                    help='change lambda\'s sign')
parser.add_argument('--epoch', default=20, type=int,
                    help='training epoch')
parser.add_argument('--aug-epoch', default=20, type=int,
                    help='attack epoch')
parser.add_argument('--train-size', default=200, type=int,
                    help='sample size')
args = parser.parse_args()
print(sys.argv)


class Oracle(object):
    def __init__(self, model, save_path='None', svm_path='None', device=None):
        self.device = device
        self.model = model
        self.save_path = save_path
        from sklearn.svm import LinearSVC
        import pickle
        with open(svm_path, 'rb') as f:
            self.svc = pickle.load(f)

    def get_loader(self, x=None, y=None, batch_size=40, shuffle=False):
        assert isinstance(x, torch.Tensor)
        if y is None:
            y = torch.full(size=(x.size(0),), fill_value=-1).long()
        dataset = torch.utils.data.TensorDataset(x, y)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=0, pin_memory=True)

    def predict(self, x, batch_size):
        self.get_loader(x, batch_size=batch_size, shuffle=False)

        preds = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            preds.append(torch.from_numpy(self.svc.predict(data.view((-1, 32 * 32 * 3)).numpy())).long())
        preds = torch.cat(preds)

        return preds

    def eval(self, x, y, batch_size):
        self.get_loader(x, y, batch_size=batch_size, shuffle=False)
        # self.model.eval()

        correct = 0
        a = time.time()

        y_true = []
        preds = []
        # outputs = np.zeros((x.size(0), n_features), dtype=np.float16)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # data = data.to(device=self.device, dtype=dtype)
                # outputs[batch_idx * batch_size:(batch_idx + 1) * batch_size] = self.model(data).cpu()

                y_true.append(target)

                # preds = torch.from_numpy(self.svc.predict(outputs)).long()
                preds.append(torch.from_numpy(
                    self.svc.predict(data.view((-1, 32 * 32 * 3)).numpy())).long())

            preds = torch.cat(preds)
            y_true = torch.cat(y_true)
            correct = preds.eq(y_true).sum().item()

            acc = correct / len(self.data_loader.dataset)
            print('Test_accuracy: %0.5f' % acc)
            print('This epoch cost %0.2f seconds' % (time.time() - a))

        return acc


class Substitute(object):

    def __init__(self, model, save_path='None', device=None):

        self.model = None

    def predict(self, x, batch_size):

        return self.model.predict(x, kind='best')

    def eval(self, x, y, batch_size):
        yp = self.model.predict(x, kind='best')
        acc = accuracy_score(y_true=y, y_pred=yp)
        print('Test_accuracy: %0.5f' % acc)
        return acc

    def train(self, x, y, batch_size, n_epoch):
        self.model = SCD(nrows=0.75, nfeatures=1, w_inc=0.17,
                         tol=0.000001, local_iter=100, num_iters=1000,
                         updated_features=128, round=1,
                         interval=20, n_jobs=1, num_gpus=1,
                         adaptive_inc=False, adv_train=False, eps=1,
                         verbose=False)
        if type(x) is torch.Tensor:
            self.model.train(x.cpu().numpy(), y.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy())
        else:
            self.model.train(x, y, x, y)


def get_data(train_size=200):
    test_dir = '/home/y/yx277/research/ImageDataset/cifar10'

    test_dataset = datasets.CIFAR10(root=test_dir, train=False,
                                    download=True,
                                    transform=None)
    data = torch.from_numpy(np.array(test_dataset.data, dtype=np.float32))
    label = torch.from_numpy(np.array(test_dataset.targets, dtype=np.int64))
    index = label < 2
    data = data[index]
    labels = label[index]

    indices = np.random.permutation(data.shape[0])
    sub_x = data[indices[:train_size]].float().reshape((-1, 32 * 32 * 3))
    sub_x /= 255
    test_data = data[indices[train_size:]].float().reshape((-1, 32 * 32 * 3))
    test_data /= 255
    test_label = labels[indices[train_size:]].long()

    return sub_x, test_data, test_label


def jacobian_augmentation(model, x_sub, y_sub, Lambda, samples_max):
    train_data = x_sub
    train_data_adv = torch.zeros_like(train_data)
    yp = torch.from_numpy(model.model.predict(train_data).reshape((-1, 1)))
    # train_data_adv[:, scd.best_w_index] -= eps * (yp * 2 - 1) * scd.best_w.numpy()
    train_data_adv[:, model.model.best_w_index] -= Lambda * (yp * 2 - 1) * 0.1
    train_data_adv += train_data
    torch.clamp(train_data_adv, 0, 1, out=train_data_adv)

    if x_sub.size(0) <= samples_max / 2:
        return torch.cat([x_sub, train_data_adv], dim=0)
    else:
        return train_data_adv


def get_adv(model, x, y, epsilon):
    train_data = x
    train_data_adv = torch.zeros_like(train_data)
    yp = torch.from_numpy(model.model.predict(train_data).reshape((-1, 1)))
    # train_data_adv[:, scd.best_w_index] -= eps * (yp * 2 - 1) * scd.best_w.numpy()
    train_data_adv[:, model.model.best_w_index] -= epsilon * (yp * 2 - 1) * 0.1
    train_data_adv += train_data
    torch.clamp(train_data_adv, 0, 1, out=train_data_adv)
    return train_data_adv


def stl10_bbox_sub(param, oracle_model, substitute_model, x_sub, test_data, \
                   test_label, aug_epoch, samples_max, n_epoch, fixed_lambda):
    clean_acc = []
    adversarial_acc = []
    for rho in range(aug_epoch):
        print('Epoch #%d:' % rho)
        # get x_sub's labels
        print('Current x_sub\'s size is %d' % (x_sub.size(0)))
        a = time.time()
        y_sub = oracle_model.predict(x=x_sub, batch_size=oracle_size)
        print('Get label for x_sub cost %.1f' % (time.time() - a))
        # train substitute model
        substitute_model.train(x=x_sub, y=y_sub, batch_size=128, n_epoch=n_epoch)
        # eval substitute on test data
        print('Substitute model evaluation on clean data: #%d:' % (test_data.size(0)))
        c = substitute_model.eval(x=test_data, y=test_label, batch_size=128)
        clean_acc.append(c)

        if rho < param['data_aug'] - 1:
            print('Substitute data augmentation processing')
            a = time.time()
            x_sub = jacobian_augmentation(model=substitute_model, x_sub=x_sub, y_sub=y_sub, \
                                          Lambda=fixed_lambda, samples_max=samples_max)
            print('Augmentation cost %.1f seconds' % (time.time() - a))

        # Generate adv examples
        test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])
        # print('Substitute model FGSM attack itself\'s accuracy on adversarial samples #%d:' % (test_adv.size(0)))
        # substitute_model.eval(x=test_adv, y=test_label, batch_size=512)
        print('Oracle model FGSM attack\'s accuracy on adversarial samples #%d:' % (test_adv.size(0)))
        b = oracle_model.eval(x=test_adv, y=test_label, batch_size=oracle_size)
        adversarial_acc.append(b)
        # torch.save(substitute_model.model.state_dict(), 'model/%s.t7' % args.target)

    return clean_acc, adversarial_acc


if __name__ == '__main__':
    param = {
        'hold_out_size': 150,
        'test_batch_size': 128,
        'nb_epochs': args.epoch,
        'learning_rate': 0.001,
        'data_aug': args.aug_epoch,
        'oracle_name': 'model/lenet',
        'epsilon': args.epsilon,
        'lambda': args.Lambda,

    }

    global seed, dtype, oracle_size, n_features
    n_features = 10000
    oracle_size = 20
    dtype = torch.float32
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0")
    device2 = torch.device("cuda:0")
    seed = 2018
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    sub_x, test_data, test_label = get_data(train_size=args.train_size)

    # oracle_model = Substitute(model=ResNet18(), save_path='stl10_checkpoint0/resnet_ckpt.t7', device=device)
    # net = torch.load('stl10_checkpoint0/rdcnn32_10000.pkl')
    # oracle_model = Oracle(model=Rh(num_layers=2, kernel_size=3, n=n_features),save_path='None',\
    #                                svm_path='model/checkpoints_svm_3_2/svm.pkl', device=device)
    oracle_model = Oracle(model=None, save_path='None', \
                          svm_path='checkpoints/%s.pkl' % args.target, device=device)
    substitute_model = Substitute(model=LinearModel(in_node=32 * 32 * 3, num_classes=2), device=device2)
    a, b = stl10_bbox_sub(param=param, oracle_model=oracle_model, substitute_model=substitute_model, \
                          x_sub=sub_x, test_data=test_data, test_label=test_label, aug_epoch=param['data_aug'], \
                          samples_max=6400, n_epoch=param['nb_epochs'], fixed_lambda=param['lambda'])

    print('\n\nFinal results:')
    print('Oracle model evaluation on clean data #%d:' % (test_data.size(0)))
    oracle_model.eval(x=test_data, y=test_label, batch_size=oracle_size)

    print('Substitute model evaluation on clean data: #%d:' % (test_data.size(0)))
    substitute_model.eval(x=test_data, y=test_label, batch_size=512)
    # test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])
    # print('Substitute model FGSM attack itself\'s accuracy on adversarial samples #%d:'%(test_adv.size(0)))
    # substitute_model.eval(x=test_adv, y=test_label, batch_size=512)
    # print('Oracle model FGSM attack\'s accuracy on adversarial samples #%d:'%(test_adv.size(0)))
    # oracle_model.eval(x=test_adv, y=test_label, batch_size=512)
    with open('../results/cifar10_/%s_%s_%s_%d.csv' %
              (args.target, str(param['epsilon']), str(param['lambda']), args.random_sign), 'w') \
            as f:
        for clean, adversarial in zip(a, b):
            f.write('%f, %f\n' % (clean, adversarial))


