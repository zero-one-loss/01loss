import numpy as np
import pickle
import sys
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
os.chdir('../data/imagenet')
train_label = np.load('train_label.npy')
test_label = np.load('test_label.npy')
train_data = np.load('train_image.npy').reshape((-1, 224*224*3)) 
test_data = np.load('test_image.npy').reshape((-1, 224*224*3)) 
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

scd = MLPClassifier(hidden_layer_sizes=(args.hidden_nodes, ), activation='logistic', solver='sgd', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=args.lr, power_t=0.5, max_iter=200, shuffle=True,
                    random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                    nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, n_iter_no_change=10)
a = time.time()
scd.fit(train, train_label)
print('Cost: %.3f seconds'%(time.time() - a))
yp = scd.predict(test)

print('Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
print('Test Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

save_path = 'checkpoints'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, args.target), 'wb') as f:
    pickle.dump(scd, f)

# class MLPEnsemble(object):
#     def __init__(self, nodes=10, numbers=32):
#         self.nodes = nodes
#         self.numbers = numbers
#         self.scd = [MLPClassifier(hidden_layer_sizes=(nodes, ), activation='logistic', solver='sgd', alpha=0.0001,
#                          batch_size='auto',
#                     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#                     random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
#                     nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#                     epsilon=1e-08, n_iter_no_change=10)] * numbers
#
#     def fit(self, train_data, train_label, ratio=0.75):
#         a = time.time()
#         n = np.arange(train_data.shape[0])
#         for i in range(self.numbers):
#             print(i)
#             # index = np.random.choice(n, int(train_data.shape[0]*ratio), False)
#             self.scd[i].fit(train_data, train_label)
#         print('Cost: %.3f seconds' % (time.time() - a))
#
#     def predict(self, test_data):
#         yp = np.zeros((test_data.shape[0], self.numbers))
#         for i in range(self.numbers):
#             yp[:, i] = self.scd[i].predict(test_data)
#         yp = yp.mean(axis=1).round()
#
#         return yp
#
#
#
# scd = MLPClassifier(20, 32)
# scd.fit(train_data, train_label)
# yp = scd.predict(test_data)
#
# print('Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train_data)))
# print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
# # yp = scd.predict(test_data, kind='vote')
# # print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
#
# save_path = 'checkpoints_svm%s'%mid_fix
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)
#
# with open(os.path.join(save_path, 'mlp_ensemble.pkl'), 'wb') as f:
#     pickle.dump(scd, f)