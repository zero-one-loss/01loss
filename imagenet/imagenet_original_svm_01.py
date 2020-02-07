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
from blackbox_attack.models import LeNet, CNNModel, LinearModel, Toy
import argparse


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
            preds.append(torch.from_numpy(self.svc.predict(data.view((-1, 224*224*3)).numpy())).long())
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
                    self.svc.predict(data.view((-1, 224*224*3)).numpy())).long())

            preds = torch.cat(preds)
            y_true = torch.cat(y_true)
            correct = preds.eq(y_true).sum().item()

            acc = correct / len(self.data_loader.dataset)
            print('Test_accuracy: %0.5f' % acc)
            print('This epoch cost %0.2f seconds' % (time.time() - a))

        return acc


class Substitute(object):

    def __init__(self, model, save_path='None', device=None):
        self.device = device
        self.model = model
        self.save_path = save_path
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(self.save_path)['net'])
            print('Load weights successfully for %s' % self.save_path)
        else:
            print('Initialized weights')
        self.model.to(device=device)

    def get_loader(self, x=None, y=None, batch_size=100, shuffle=False):
        assert isinstance(x, torch.Tensor)
        if y is None:
            y = torch.full(size=(x.size(0),), fill_value=-1).long()
        dataset = torch.utils.data.TensorDataset(x, y)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=2, pin_memory=True)

    def predict(self, x, batch_size):
        self.get_loader(x, batch_size=batch_size, shuffle=False)
        self.model.eval()
        pred = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data = data.to(device=self.device, dtype=dtype)

                outputs = self.model(data)
                pred.append(outputs.data.max(1)[1])

        return torch.cat(pred).cpu()

    def eval(self, x, y, batch_size):
        self.get_loader(x, y, batch_size=batch_size, shuffle=False)
        self.model.eval()

        correct = 0
        a = time.time()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.data_loader):

                data, target = data.to(device=self.device, dtype=dtype), target.to(device=self.device)

                predicted = self.model(data).max(1)[1]
                # outputs = self.model(data)
                # predicted = outputs.data.max(1)[1]
                correct += predicted.eq(target.data).sum().item()

            acc = correct / len(self.data_loader.dataset)
            print('Test_accuracy: %0.5f' % acc)
            # print('This epoch cost %0.2f seconds' % (time.time() - a))

            return acc

    def train(self, x, y, batch_size, n_epoch):
        self.get_loader(x, y, batch_size, True)
        self.model.train()

        optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=args.lr,
                )
        criterion = nn.CrossEntropyLoss().to(device=self.device)
        for epoch in range(n_epoch):
            train_loss = 0
            correct = 0
            a = time.time()

            for batch_idx, (data, target) in enumerate(self.data_loader):

                data, target = data.to(device=self.device, dtype=dtype), target.to(device=self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = outputs.data.max(1)[1]
                correct += predicted.eq(target.data).sum().item()

            # print('Epoch #%d'%epoch)
            # print('Train loss: %0.5f,     Train_accuracy: %0.5f' % (
            #     train_loss / len(self.data_loader.dataset), correct / len(self.data_loader.dataset)))
            # print('This epoch cost %0.2f seconds' % (time.time() - a))

    def get_grad(self, x, y):
        self.get_loader(x, y, batch_size=1, shuffle=False)
        grads = []
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(device=self.device, dtype=dtype).requires_grad_()
            outputs = self.model(data)[0, target]
            outputs.backward()
            grads.append(data.grad.cpu())
        return torch.cat(grads, dim=0)

    def get_loss_grad(self, x, y):
        self.get_loader(x, y, batch_size=100, shuffle=False)
        grads = []

        self.model.train()
        criterion = nn.CrossEntropyLoss().to(device=self.device)
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(device=self.device, dtype=dtype).requires_grad_(),\
                           target.to(device=self.device)
            outputs = self.model(data)
            loss = criterion(outputs, target)
            loss.backward()
            grads.append(data.grad.cpu())
        return torch.cat(grads, dim=0)

def get_data(train_size=200):

    test_dir = '../data/imagenet'

    data = torch.from_numpy(np.load('%s/test_image.npy' % test_dir))
    label = torch.from_numpy(np.load('%s/test_label.npy' % test_dir))
    index = label < 2
    data = data[index]
    labels = label[index]

    indices = np.random.permutation(data.shape[0])
    sub_x = data[indices[:train_size]].float()
    # sub_x /= 255
    test_data = data[indices[train_size:]].float()
    # test_data /= 255
    test_label = labels[indices[train_size:]].long()

    return sub_x, test_data, test_label

# def get_train_data():
#     test_dir = '/home/y/yx277/research/ImageDataset/mnist'
#
#     train_dataset = datasets.MNIST(root=test_dir, train=True,
#                                   download=True,
#                                   transform=None)
#     test_dataset = datasets.MNIST(root=test_dir, train=False,
#                                   download=True,
#                                   transform=None)
#     data = train_dataset.train_data
#     label = train_dataset.train_labels
#     index = label < 2
#     data = data[index]
#     labels = label[index]
#
#     train_data = data.float().reshape((-1, 1, 28, 28))
#     train_labels = labels.long()
#
#     data = test_dataset.test_data
#     label = test_dataset.test_labels
#     index = label < 2
#     data = data[index]
#     labels = label[index]
#
#     test_data = data.float().reshape((-1, 1, 28, 28))
#     test_labels = labels.long()
#     return train_data, train_labels, test_data, test_labels

def jacobian_augmentation(model, x_sub, y_sub, Lambda, samples_max):
    if args.random_sign == 1:
        Lambda = np.random.choice([-1, 1])* Lambda
    x_sub_grads = model.get_grad(x=x_sub, y=y_sub)
    x_sub_new = x_sub + Lambda * torch.sign(x_sub_grads)
    if x_sub.size(0) <= samples_max / 2:
        return torch.cat([x_sub, x_sub_new], dim=0)
    else:
        return x_sub_new

def get_adv(model, x, y, epsilon):
    print('getting grads on epsilon=%.4f'%epsilon)
    grads = model.get_loss_grad(x, y)
    print('generating adversarial examples')
    return (x + epsilon * torch.sign(grads)).clamp_(0, 1)


def imagenet_bbox_sub(param, oracle_model, substitute_model, x_sub, test_data, \
                   test_label, aug_epoch, samples_max, n_epoch, fixed_lambda):
    clean_acc = []
    adversarial_acc = []
    for rho in range(aug_epoch):
        print('Epoch #%d:'%rho)
        # get x_sub's labels
        print('Current x_sub\'s size is %d'%(x_sub.size(0)))
        a = time.time()
        y_sub = oracle_model.predict(x=x_sub, batch_size=oracle_size)
        print('Get label for x_sub cost %.1f'%(time.time() - a))
        #train substitute model
        substitute_model.train(x=x_sub, y=y_sub, batch_size=128, n_epoch=n_epoch)
        #eval substitute on test data
        print('Substitute model evaluation on clean data: #%d:' % (test_data.size(0)))
        c = substitute_model.eval(x=test_data, y=test_label, batch_size=128)
        clean_acc.append(c)

        if rho < param['data_aug'] - 1:
            print('Substitute data augmentation processing')
            a = time.time()
            x_sub = jacobian_augmentation(model=substitute_model, x_sub=x_sub, y_sub=y_sub, \
                                          Lambda=fixed_lambda, samples_max=samples_max)
            print('Augmentation cost %.1f seconds'%(time.time() - a))


        #Generate adv examples
        test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])
        # print('Substitute model FGSM attack itself\'s accuracy on adversarial samples #%d:' % (test_adv.size(0)))
        # substitute_model.eval(x=test_adv, y=test_label, batch_size=512)
        print('Oracle model FGSM attack\'s accuracy on adversarial samples #%d:' % (test_adv.size(0)))
        b = oracle_model.eval(x=test_adv, y=test_label, batch_size=oracle_size)
        adversarial_acc.append(b)
        torch.save(substitute_model.model.state_dict(), 'model/%s.t7' % args.target)

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

    # oracle_model = Substitute(model=ResNet18(), save_path='imagenet_checkpoint0/resnet_ckpt.t7', device=device)
    # net = torch.load('imagenet_checkpoint0/rdcnn32_10000.pkl')
    # oracle_model = Oracle(model=Rh(num_layers=2, kernel_size=3, n=n_features),save_path='None',\
    #                                svm_path='model/checkpoints_svm_3_2/svm.pkl', device=device)
    oracle_model = Oracle(model=None, save_path='None',\
                                   svm_path='checkpoints/%s.pkl' % args.target, device=device)
    substitute_model = Substitute(model=LinearModel(in_node=224*224*3,num_classes=2), device=device2)
    a, b = imagenet_bbox_sub(param=param, oracle_model=oracle_model, substitute_model=substitute_model, \
                   x_sub=sub_x, test_data=test_data, test_label=test_label, aug_epoch=param['data_aug'],\
                   samples_max=6400, n_epoch=param['nb_epochs'], fixed_lambda=param['lambda'])

    print('\n\nFinal results:')
    print('Oracle model evaluation on clean data #%d:'%(test_data.size(0)))
    oracle_model.eval(x=test_data, y=test_label, batch_size=oracle_size)

    print('Substitute model evaluation on clean data: #%d:'%(test_data.size(0)))
    substitute_model.eval(x=test_data, y=test_label, batch_size=512)
    # test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])
    # print('Substitute model FGSM attack itself\'s accuracy on adversarial samples #%d:'%(test_adv.size(0)))
    # substitute_model.eval(x=test_adv, y=test_label, batch_size=512)
    # print('Oracle model FGSM attack\'s accuracy on adversarial samples #%d:'%(test_adv.size(0)))
    # oracle_model.eval(x=test_adv, y=test_label, batch_size=512)
    with open('../results/imagenet/%s_%s_%s_%d.csv' %
              (args.target, str(param['epsilon']), str(param['lambda']), args.random_sign), 'w') \
        as f:
        for clean, adversarial in zip(a, b):
            f.write('%f, %f\n' % (clean, adversarial))
            
    # train_data, train_label, test_data, test_label = get_train_data()
    #
    # train_adv = get_adv(model=substitute_model, x=train_data, y=train_label, epsilon=param['epsilon'])
    # test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])
    # train_adv = torch.cat([train_data, train_adv]).squeeze(1).numpy().reshape((-1, 224*224*3))
    # train_labels = torch.cat([train_label, train_label]).numpy()
    #
    # test_adv = torch.cat([test_data, test_adv]).squeeze(1).numpy().reshape((-1, 224*224*3))
    # test_labels = torch.cat([test_label, test_label]).numpy()
    #
    # os.chdir('mnist_adv_data')
    # np.save('train_data_%s.npy' % str(param['epsilon']), train_adv)
    # np.save('train_label_%s.npy' % str(param['epsilon']), train_labels)
    # np.save('test_data_%s.npy' % str(param['epsilon']), test_adv)
    # np.save('test_label_%s.npy' % str(param['epsilon']), test_labels)
    # os.chdir(os.pardir)

