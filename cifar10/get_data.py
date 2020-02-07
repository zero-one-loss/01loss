import torchvision
import numpy as np
import torchvision.transforms as transforms
import os

DATA = 'cifar10'

if DATA == 'cifar10':
    train_dir = '/home/y/yx277/research/ImageDataset/cifar10'
    test_dir = '/home/y/yx277/research/ImageDataset/cifar10'


test_transform = transforms.Compose(
        [
                transforms.ToTensor(),
                ])

trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=False, transform=test_transform)

testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=False, transform=test_transform)

save_path = '../data/cifar10'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

os.chdir(save_path)
np.save('train_image.npy', trainset.data)
np.save('train_label.npy', np.array(trainset.targets))
np.save('test_image.npy', testset.data)
np.save('test_label.npy', np.array(testset.targets))
os.chdir(os.pardir)

