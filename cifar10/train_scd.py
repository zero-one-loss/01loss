import numpy as np
import pickle
import sys
sys.path.append('..')
from core.scd_v9_gpu import SCD
import time
import os
from sklearn.metrics import accuracy_score
import argparse


parser = argparse.ArgumentParser(description='SCD 01 loss')

parser.add_argument('--nrows', default=0.75, type=float,
                    help='ratio of rows in each iteration')
parser.add_argument('--nfeatures', default=1, type=float,
                    help='ratio of features in each vote')
parser.add_argument('--w-inc', default=0.17, type=float,
                    help='weights increments')
parser.add_argument('--num-iters', default=100, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--updated-features', default=128, type=int,
                    help='number of features will be update in each iteration')
parser.add_argument('--round', default=1, type=int,
                    help='number of vote')
parser.add_argument('--interval', default=10, type=int,
                    help='number of neighbours will be considered '
                         'in bias choosing')
parser.add_argument('--n-jobs', default=2, type=int,
                    help='number of processes')
parser.add_argument('--num-gpus', default=1, type=int,
                    help='number of GPUs')
parser.add_argument('--adv-train', action='store_true',
                    help='Run adversarail training')
parser.add_argument('--eps', default=1, type=float,
                    help='epsilon in adversarial training')
parser.add_argument('--verbose', action='store_true',
                    help='show intermediate acc output')
parser.add_argument('--target', default='scd.pkl', type=str,
                    help='checkpoint\'s name')
args = parser.parse_args()

adv_train = True if args.adv_train else False
verbose = True if args.verbose else False

# Load data
curdir = os.getcwd()
os.chdir('../data/cifar10')
train_label = np.load('train_label.npy')
test_label = np.load('test_label.npy')
train_data = np.load('train_image.npy').reshape((-1, 32*32*3)) / 255
test_data = np.load('test_image.npy').reshape((-1, 32*32*3)) / 255
os.chdir(curdir)

# pick first two classes
np.random.seed(2018)
train = train_data[train_label < 2]
test = test_data[test_label < 2]
train_label = train_label[train_label < 2]
test_label = test_label[test_label < 2]
print('training data size: ')
print(train.shape)
print('testing data size: ')
print(test.shape)

# initialize scd instance

scd = SCD(nrows=args.nrows, nfeatures=args.nfeatures, w_inc=args.w_inc,
          tol=0.000001, local_iter=100, num_iters=args.num_iters,
          updated_features=args.updated_features, round=args.round,
          interval=args.interval, n_jobs=args.n_jobs, num_gpus=args.num_gpus,
          adaptive_inc=False, adv_train=adv_train, eps=args.eps,
          verbose=verbose)

a = time.time()
scd.train(train, train_label, test, test_label)
print('Cost: %.3f seconds'%(time.time() - a))
yp = scd.predict(test)

print('Best Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
print('Vote Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train, kind='vote')))
print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
yp = scd.predict(test, kind='vote')
print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

save_path = 'checkpoints'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, '%s' % args.target), 'wb') as f:
    pickle.dump(scd, f)

