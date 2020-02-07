import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from resnet import ResNet18
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
# from util.misc import CSVLogger



DATA = 'cifar10'

if DATA == 'cifar10':
    train_dir = '/home/y/yx277/research/ImageDataset/cifar10'
    test_dir = '/home/y/yx277/research/ImageDataset/cifar10'



resume = True
use_cuda = True
dtype = torch.float32
if sys.argv[1] == '0':
    aug = 'noaug'
else:
    aug = 'aug'

best_acc = 0

batch_size = 512

seed = 2018
print('Random seed: ', seed)
torch.manual_seed(seed)
save_path = 'resnet18_%s_checkpoint'%aug
if not os.path.isdir('logs'):
    os.mkdir('logs')
filename = 'logs/resnet18_%s.csv'%aug
# csv_logger = CSVLogger(fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

train_transform = transforms.Compose(
                [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ])

test_transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ])

if aug == 'noaug':
    train_transform = test_transform

print('start normalize')

trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=train_transform)
index = [i for i in range(len(trainset)) if trainset.targets[i] < 2]
trainset.data = trainset.data[index]
p = np.array(trainset.targets)
trainset.targets = p[index].tolist()
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=test_transform)
index = [i for i in range(len(testset)) if testset.targets[i] < 2]
testset.data = testset.data[index]
p = np.array(testset.targets)
testset.targets = p[index].tolist()
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

net = ResNet18(2)

criterion = nn.CrossEntropyLoss()

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/resnet_ckpt.t7')
    net.load_state_dict(checkpoint['net'])

if use_cuda:
    print('start move to cuda')
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    # net = net.half()
    # for layer in net.modules():
    #     if isinstance(layer, nn.BatchNorm2d):
    #         layer.float()
    # net = torch.nn.DataParallel(net, device_ids=[0,1])
    device = torch.device("cuda:0")
    net.to(device=device)
    criterion.to(device=device, dtype=dtype)




optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True
)


def train(epoch):
    # global monitor
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    a = time.time()
    #    pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.max(1)[1]
        correct += predicted.eq(target).sum().item()

    print('Train loss: %0.5f,     Train_accuracy: %0.5f' % (
        train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)))
    print('This epoch cost %0.2f seconds' % (time.time() - a))

    return correct / len(train_loader.dataset)


def test(epoch):
    global best_acc
    # monitor
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            outputs= net(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            predicted = outputs.max(1)[1]
            correct += predicted.eq(target).sum().item()

        print('Test loss: %0.5f,     Test_accuracy: %0.5f' % (
            test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
        print('This epoch cost %0.2f seconds' % (time.time() - a))

    acc = correct / len(test_loader.dataset)
    if acc > best_acc:
        print('Saving...')
        state = {
            # 'net': net.module.state_dict(),
            'net': net.state_dict(),
        }

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/ckpt.t7')
        best_acc = acc

    return acc

def save_features():
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()

    os.chdir('temp_data')

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    with torch.no_grad():
        for i in range(10):
            for batch_idx, (data, target) in enumerate(train_loader):
                if use_cuda:
                    data = data.to(device=device, dtype=dtype)

                features = net(data, out_features=True)
                train_features.append(features.data.cpu())
                train_labels.append(target)
    train_features = torch.cat(train_features, dim=0).numpy()
    train_labels = torch.cat(train_labels, dim=0).numpy()
    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    
    
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data = data.to(device=device, dtype=dtype)

            features = net(data, out_features=True)
            test_features.append(features.data.cpu())
            test_labels.append(target)
    test_features = torch.cat(test_features, dim=0).numpy()
    test_labels = torch.cat(test_labels, dim=0).numpy()
    np.save('test_features.npy', test_features)
    np.save('test_labels.npy', test_labels)
    os.chdir(os.pardir)

def main():
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + 80):

        train(epoch)
        test(epoch)

        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])
        if epoch in [50, 70]:
            optimizer.param_groups[0]['lr'] *= 0.1



if __name__ == '__main__':
    save_features()

