import numpy as np
import pickle
import sys
sys.path.append('..')
from core.mlp_ensemble import MLPEnsemble
import time
import os
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


import argparse

parser = argparse.ArgumentParser(description='mlp')

parser.add_argument('--hidden-nodes', default=20, type=int,
                    help='number of hidden nodes')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--iters', default=200, type=int,
                    help='ratio of rows in each iteration')
parser.add_argument('--target', default='mlp.pkl', type=str,
                    help='checkpoint\'s name')
args = parser.parse_args()

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

np.random.seed(2018)

# scd = MLPClassifier(hidden_layer_sizes=(args.hidden_nodes, ), activation='logistic', solver='sgd', alpha=0.0001, batch_size='auto',
#                     learning_rate='constant', learning_rate_init=args.lr, power_t=0.5, max_iter=200, shuffle=True,
#                     random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
#                     nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#                     epsilon=1e-08, n_iter_no_change=10)
# a = time.time()
# scd.fit(train, train_label)
# print('Cost: %.3f seconds'%(time.time() - a))
# yp = scd.predict(test)
#
# print('Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
# print('Test Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
#
# save_path = 'checkpoints'
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)
#
# with open(os.path.join(save_path, args.target), 'wb') as f:
#     pickle.dump(scd, f)




scd = MLPEnsemble(nodes=args.hidden_nodes, numbers=32,
                  lr=args.lr, max_iter=args.iters)
scd.fit(train, train_label)
yp = scd.predict(test)

save_path = 'checkpoints'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, args.target), 'wb') as f:
    pickle.dump(scd, f)

print('Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
# yp = scd.predict(test_data, kind='vote')
# print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

