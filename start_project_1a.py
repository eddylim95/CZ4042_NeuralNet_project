#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
# import pylab as plt
import matplotlib.pyplot as plt
import sys

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# Build the graph for the deep net
def ffn(x, num_neurons):
    """Feedforward net with 1 hidden layer
    """
    sum_regularization = 0
    with tf.name_scope('hidden'):
        weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
        biases  = tf.Variable(tf.zeros([num_neurons]), name='biases')
        h  = tf.nn.relu(tf.matmul(x, weights) + biases)
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    with tf.name_scope('linear'):
        weights = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
        biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        u = tf.matmul(h, weights) + biases
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    
    return u, sum_regularization

def make_train_model(X,Y,batch_size, num_neurons_list):

    train_acc = []

    test_split_num = int(len(trainX) * test_split)
    train_x, test_x = X[:test_split_num], X[test_split_num:]
    train_y, test_y = Y[:test_split_num], Y[test_split_num:]
    for fold in range(num_folds):
        print(f'Fold number: {fold+1}')
        n = int(train_x.shape[0] / num_folds)
        fold_start, fold_end = fold*n, (fold+1)*n
        x_test, y_test = test_x[fold_start:fold_end], test_y[fold_start:fold_end]
        x_train  = np.append(train_x[:fold_start], train_x[fold_end:], axis=0)
        y_train = np.append(train_y[:fold_start], train_y[fold_end:], axis=0) 
        # print(test_index)

        train_acc_ = []
        for num_neurons in num_neurons_list:
            # Create the model
            print(f'Training {num_neurons} for hidden layer')
            x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
            y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

            logits, regularizer = ffn(x, num_neurons)

            cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
            loss = tf.reduce_mean(cost + weight_decay_beta*regularizer)

            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)

            correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(epochs):
                    # Handle in batches
                    for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
                        train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
                    train_acc_.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

                    if i % 100 == 0:
                        print('iter %d: accuracy %g'%(i, train_acc_[i]))
        train_acc.append(train_acc_)
    return train_acc

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

NUM_FEATURES = 21
NUM_CLASSES = 3

num_folds = 5
learning_rate = 0.01
epochs = 5000
weight_decay_beta = 0.000001
batch_size = [4, 8, 16, 32, 64]
num_neurons_list = [5,10,15,20,25]
seed = 10
test_split = 0.3
np.random.seed(seed)

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 # one hot matrix

# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

train_acc = make_train_model(trainX, trainY ,batch_size[3], [num_neurons_list[1]]) # Q1, batch size 32, 10 hidden neurons

fig, ax = plt.subplots()
for i in range(len(train_acc)):
    ax.plot(range(epochs), train_acc[i], linewidth=2, label=f'neuron: {num_neurons_list[1]}, batch_size: {batch_size[3]}, fold: {i+1}')
ax.legend(loc='lower right')
plt.savefig('plots/part1_10_32.png')
plt.show()
