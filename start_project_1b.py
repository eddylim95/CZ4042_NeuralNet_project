#%%
#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def ffn(x, neuron_size, weight_decay_beta, layers=3):
    """Feedforward net with 1 hidden layer
    """
    sum_regularization = 0
    with tf.name_scope('hidden'):
        weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, neuron_size], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.zeros([neuron_size]), dtype=tf.float32, name='biases')
        h  = tf.nn.relu(tf.matmul(x, weights) + biases)
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    if layers > 3:
        for i in range(layers-3):
            with tf.name_scope('hidden{}'.format(i)):
                weights = tf.Variable(tf.truncated_normal([neuron_size, neuron_size], stddev=1.0 / np.sqrt(neuron_size), dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([neuron_size]), dtype=tf.float32, name='biases')
                h  = tf.nn.relu(tf.matmul(h, weights) + biases)
                sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    with tf.name_scope('linear'):
        weights = tf.Variable(tf.truncated_normal([neuron_size, 1], stddev=1.0 / np.sqrt(neuron_size), dtype=tf.float32), name='weights')
        biases  = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
        u = tf.matmul(h, weights) + biases
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)


    return u, sum_regularization

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
trainX = X_data[:50]
trainY = Y_data[:50]

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

test_split_num = int(len(trainX) * test_split)
train_x, test_x = trainX[:test_split_num], trainX[test_split_num:]
train_y, test_y = trainY[:test_split_num], trainY[test_split_num:]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

y, regularizer = ffn(x, neuron_size, weight_decay_beta)

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
cost = tf.square(y_ - y)
loss = tf.reduce_mean(cost + regularizer)
train_op = optimizer.minimize(loss)

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
    for i in range(50):
        pred = train_op.run(loss, feed_dict={x: X_data[:-50]})
        prediction.append(pred)
#%%
print(np.array(prediction))
#%%
# plot learning curves
fig, ax = plt.subplots()
# plt.figure(1)
plt.plot(range(epochs), train_err, label=f'Train Error')
plt.plot(range(epochs), test_err, label=f'Test Error')
plt.xlabel(str(epochs) + ' epochs')
plt.ylabel('Mean Square Error')
ax.legend(loc='best')
plt.savefig('plots2/part2_Q1a')
plt.show()
#%%
fig, ax = plt.subplots()
# plt.figure(1)
plt.plot(range(50), prediction[0], label=f'Prediction')
plt.plot(range(50), Y_data[:-50], label=f'Actual')
plt.xlabel(str(50) + ' epochs')
plt.ylabel('Prediction')
ax.legend(loc='best')
# plt.savefig('plots2/part2_Q1c')
plt.show()

#%%
# Q2a
import pandas as pd
df = pd.read_csv('admission_predict.csv')
df = df.iloc[:,1:]
df = df.corr()
df.to_csv('plots2/correlation_matrix.csv')
#%%
