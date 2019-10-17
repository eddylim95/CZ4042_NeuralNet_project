#%%
#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def ffn(x, feature_size, neuron_size, weight_decay_beta, layers=3, dropout=False):
    """Feedforward net with hidden layers
    """
    sum_regularization = 0
    with tf.name_scope('hidden'):
        weights = tf.Variable(tf.truncated_normal([feature_size, neuron_size], stddev=1.0 / np.sqrt(feature_size), dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.zeros([neuron_size]), dtype=tf.float32, name='biases')
        h  = tf.nn.relu(tf.matmul(x, weights) + biases)
        if dropout:
            h = tf.nn.dropout(h, 0.8)
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    if layers > 3:
        for i in range(layers-3):
            with tf.name_scope('hidden{}'.format(i)):
                weights = tf.Variable(tf.truncated_normal([neuron_size, neuron_size], stddev=1.0 / np.sqrt(neuron_size), dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([neuron_size]), dtype=tf.float32, name='biases')
                h  = tf.nn.relu(tf.matmul(h, weights) + biases)
                if dropout:
                    h = tf.nn.dropout(h, 0.8)
                sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    with tf.name_scope('linear'):
        weights = tf.Variable(tf.truncated_normal([neuron_size, 1], stddev=1.0 / np.sqrt(neuron_size), dtype=tf.float32), name='weights')
        biases  = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
        u = tf.matmul(h, weights) + biases
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)


    return u, sum_regularization

def create_model(feature_size, neuron_size, weight_decay_beta, learning_rate, layers=3, dropout=False):
    # Create the model
    x = tf.placeholder(tf.float32, [None, feature_size])
    y_ = tf.placeholder(tf.float32, [None, 1])
    y, regularizer = ffn(x, feature_size, neuron_size, weight_decay_beta, layers=layers, dropout=dropout)

    #Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    cost = tf.square(y_ - y)
    loss = tf.reduce_mean(cost + regularizer)
    train_op = optimizer.minimize(loss)
    return y, train_op, y_, x, loss

def train_model(train_op, train_x, train_y, test_x, test_y, y, y_, x, loss ,sample_X=[]):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        test_err = []
        prediction = []
        for i in range(epochs):
            # Handle in batches
            for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x), batch_size)):
                train_op.run(feed_dict={x: train_x[start:end], y_: train_y[start:end]})
            err = loss.eval(feed_dict={x: train_x, y_: train_y})
            test_err_ = loss.eval(feed_dict={x: test_x, y_: test_y})
            train_err.append(err)
            test_err.append(test_err_)

            if i % 100 == 0:
                print('iter %d: train error %g'%(i, train_err[i]))
        if sample_X != []:
            prediction = sess.run(y, feed_dict={x: sample_X})
    return test_err, train_err, prediction

def plot_rfe_loss(filename, x_headers, epochs, plot_list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    # plt.figure(1)
    for i, plot in enumerate(plot_list):
        plt.plot(range(epochs), plot[-epochs:], label=f'Without {x_headers[i]}')
    plt.xlabel(str(epochs) + ' epochs')
    plt.ylabel('Mean Square Error')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_test_err_comparison(filename, epochs, error_list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    for i,errors in enumerate(error_list):
        plt.plot(range(epochs), errors[-epochs:], label=f'Test Error with {i} RFE')
    plt.xlabel(str(epochs) + ' epochs')
    plt.ylabel('Mean Square Error')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_acc_vs_pred(filename, prediction, Y_data):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    plt.plot(range(50), prediction, label=f'Prediction')
    plt.plot(range(50), Y_data[-50:], label=f'Actual')
    plt.xlabel(str(50) + ' epochs')
    plt.ylabel('Prediction')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_train_test_err(filename, epochs, train_err, test_err):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    plt.plot(range(epochs), train_err, label=f'Train Error', color='green')
    plt.plot(range(epochs), test_err, label=f'Test Error', color='red')
    plt.xlabel(str(epochs) + ' epochs')
    plt.ylabel('Mean Square Error')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_layer_comp(filename, epochs, err_list, train_or_test):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    for i in range(3):
        plt.plot(range(epochs), err_list[2*i], label=f'{train_or_test} {i+3}-layer net without dropout')
        plt.plot(range(epochs), err_list[2*i+1], label=f'{train_or_test} {i+3}-layer net with dropout')
    plt.xlabel(str(epochs) + ' epochs')
    plt.ylabel('Mean Square Error')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

# Initial base parameters
NUM_FEATURES = 7

learning_rate = 0.01
epochs = 1000
batch_size = 8
neuron_size = 30
weight_decay_beta = float('10e-3')
seed = 10
test_split = 0.3
np.random.seed(seed)

#read and divide data into test and train sets 
admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
Y_data = Y_data.reshape(Y_data.shape[0], 1)

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

# experiment with small datasets
# trainX = X_data[:100]
# trainY = Y_data[:100]
trainX = X_data
trainY = Y_data

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

test_split_num = int(len(trainX) * test_split)
train_x, test_x = trainX[test_split_num:], trainX[:test_split_num]
train_y, test_y = trainY[test_split_num:], trainY[:test_split_num]

sample_X = trainX[-50:]

#%%
# Q1
y, train_op, y_, x, loss = create_model(NUM_FEATURES, neuron_size, weight_decay_beta, learning_rate)
test_err, train_err, prediction = train_model(train_op, train_x, train_y, test_x, test_y, y, y_, x, loss, sample_X)
#%%
plot_train_test_err('plots2/part2_Q1a', epochs, train_err, test_err)
plot_acc_vs_pred('plots2/part2_Q1c', prediction, Y_data)

#%%
# Q2a
df = pd.read_csv('admission_predict.csv')
df = df.iloc[:,1:]
df = df.corr()
df.to_csv('plots2/correlation_matrix.csv')

#%%
# Q3p1
y, train_op, y_, x, loss = create_model(6, neuron_size, weight_decay_beta, learning_rate)
test_err_list = []
train_err_list = []
prediction_list = []
x_headers = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']

for i in range(7):
    if i == 0:
        train_x_ = train_x[:, i+1:]
        test_x_ = test_x[:, i+1:]
        print(f'With {x_headers[i+1:]}')
    elif i == 6:
        train_x_ = train_x[:, :i]
        test_x_ = test_x[:, :i]
        print(f'With {x_headers[:i]}')
    else:
        train_x_ = np.append(train_x[:, :i], train_x[:, i+1:], axis=1)
        test_x_ = np.append(test_x[:, :i], test_x[:, i+1:], axis=1)
        print(f'With {np.append(x_headers[:i], x_headers[i+1:], axis=0)}')
    test_err, train_err, prediction = train_model(train_op, train_x_, train_y, test_x_, test_y, y, y_, x, loss)
    test_err_list.append(test_err)
    train_err_list.append(train_err)
    prediction_list.append(prediction)

#%%
# Conclusion: Remove University Ranking
plot_rfe_loss('plots2/part3_1', x_headers, epochs, train_err_list)
plot_rfe_loss('plots2/part3_2', x_headers, epochs, test_err_list)
plot_rfe_loss('plots2/part3_3', x_headers, 100, test_err_list)

#%%
# Q3p2. Remove University Ranking
y, train_op, y_, x, loss = create_model(5, neuron_size, weight_decay_beta, learning_rate)
test_err_list = []
train_err_list = []
prediction_list = []
x_headers2 = ['GRE Score','TOEFL Score','SOP','LOR','CGPA','Research']

for i in range(6):
    # Remove University Ranking
    train_x_ = np.append(train_x[:, :2], train_x[:, 2+1:], axis=1)
    test_x_ = np.append(test_x[:, :2], test_x[:, 2+1:], axis=1)
    if i == 0:
        train_x_2 = train_x_[:, i+1:]
        test_x_2 = test_x_[:, i+1:]
        print(f'With {x_headers2[i+1:]}')
    elif i == 6:
        train_x_2 = train_x_[:, :i]
        test_x_2 = test_x_[:, :i]
        print(f'With {x_headers2[:i]}')
    else:
        train_x_2 = np.append(train_x_[:, :i], train_x_[:, i+1:], axis=1)
        test_x_2 = np.append(test_x_[:, :i], test_x_[:, i+1:], axis=1)
        print(f'With {np.append(x_headers2[:i], x_headers2[i+1:], axis=0)}')
    test_err, train_err, prediction = train_model(train_op, train_x_2, train_y, test_x_2, test_y, y, y_, x, loss)
    test_err_list.append(test_err)
    train_err_list.append(train_err)
    prediction_list.append(prediction)

#%%
# Conclusion: Remove SOP
plot_rfe_loss('plots2/part3_4', x_headers2, epochs, train_err_list)
plot_rfe_loss('plots2/part3_5', x_headers2, epochs, test_err_list)
plot_rfe_loss('plots2/part3_6', x_headers2, 100, test_err_list)

#%%
# Q3 comparison between RFE
test_err_list = []
train_err_list = []
prediction_list = []

# Before any removal
y, train_op, y_, x, loss = create_model(7, neuron_size, weight_decay_beta, learning_rate)
test_err, train_err, prediction = train_model(train_op, train_x, train_y, test_x, test_y, y, y_, x, loss)
test_err_list.append(test_err)
train_err_list.append(train_err)
prediction_list.append(prediction)

# Remove University Ranking
train_x_ = np.append(train_x[:, :2], train_x[:, 2+1:], axis=1)
test_x_ = np.append(test_x[:, :2], test_x[:, 2+1:], axis=1)

y, train_op, y_, x, loss = create_model(6, neuron_size, weight_decay_beta, learning_rate)
test_err, train_err, prediction = train_model(train_op, train_x_, train_y, test_x_, test_y, y, y_, x, loss)
test_err_list.append(test_err)
train_err_list.append(train_err)
prediction_list.append(prediction)

# Remove SOP
train_x_ = np.append(train_x_[:, :2], train_x_[:, 2+1:], axis=1)
test_x_ = np.append(test_x_[:, :2], test_x_[:, 2+1:], axis=1)

y, train_op, y_, x, loss = create_model(5, neuron_size, weight_decay_beta, learning_rate)
test_err, train_err, prediction = train_model(train_op, train_x_, train_y, test_x_, test_y, y, y_, x, loss)
test_err_list.append(test_err)
train_err_list.append(train_err)
prediction_list.append(prediction)

#%%
plot_test_err_comparison('plots2/part3_7', epochs, test_err_list)
plot_test_err_comparison('plots2/part3_8', 100, test_err_list)

#%%
# Q4. Neuron size 50, 4 and 5 layer network, learning rate 10e-3, 
# features = ['GRE Score','TOEFL Score','LOR','CGPA','Research']
test_err_list = []
train_err_list = []
prediction_list = []

# Remove University Ranking
train_x_ = np.append(train_x[:, :2], train_x[:, 2+1:], axis=1)
test_x_ = np.append(test_x[:, :2], test_x[:, 2+1:], axis=1)
# Remove SOP
train_x_ = np.append(train_x_[:, :2], train_x_[:, 2+1:], axis=1)
test_x_ = np.append(test_x_[:, :2], test_x_[:, 2+1:], axis=1)

for i in range(3, 6):
    # No Dropouts
    y, train_op, y_, x, loss = create_model(5, neuron_size, weight_decay_beta, learning_rate, layers=i)

    test_err, train_err, prediction = train_model(train_op, train_x_, train_y, test_x_, test_y, y, y_, x, loss)
    test_err_list.append(test_err)
    train_err_list.append(train_err)
    prediction_list.append(prediction)

    # With Dropouts
    y, train_op, y_, x, loss = create_model(5, neuron_size, weight_decay_beta, learning_rate, layers=i, dropout=True)

    test_err, train_err, prediction = train_model(train_op, train_x_, train_y, test_x_, test_y, y, y_, x, loss)
    test_err_list.append(test_err)
    train_err_list.append(train_err)
    prediction_list.append(prediction)

#%%
# 3-layer net without dropout is the best
plot_layer_comp('plots2/part4_1', epochs, test_err_list, 'Test')
plot_layer_comp('plots2/part4_2', epochs, train_err_list, 'Train')
