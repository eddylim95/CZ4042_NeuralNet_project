#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def batch_separator(batch_size, data_len):
    """Separates into batches
    """
    batch_mask = np.random

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 5000
weight_decay = 0.000001
batch_size = [4, 8, 16, 32, 64]
NUM_NEURONS = [5,10,15,20,25]
seed = 10
train_split = int(len(trainX) * 7/10)
np.random.seed(seed)

# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 # one hot matrix
train_x, val_x = trainX[:train_split], trainX[train_split:]
train_y, val_y = trainY[:train_split], trainY[train_split:]
n = train_x.shape[0]

def make_train_model(batch_size, num_neurons):
    # Reset the graph
    tf.reset_default_graph()

    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    w1 = tf.Variable(tf.truncated_normal(shape=[NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
    b1  = tf.Variable(tf.zeros([num_neurons]), name='biases')
    x1  = tf.sigmoid(tf.matmul(x, w1) + b1)

    # Hidden layer with relu
    w2 = tf.Variable(tf.truncated_normal(shape=[num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
    b2  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    h1 = tf.matmul(x1, w2) + b2
    logits = tf.nn.relu(h1)

    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cost)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_acc = []
        for i in range(epochs):
            train_op.run(feed_dict={x: train_x, y_: train_y})
            train_acc.append(accuracy.eval(feed_dict={x: val_x, y_: val_y}))

            # weights1 = sess.run(w1)
            # bias1 = sess.run(b1)
            # weights2 = sess.run(w2)
            # bias2 = sess.run(b2)

            if i % 100 == 0:
                print('iter %d: accuracy %g'%(i, train_acc[i]))
    return train_acc

train_acc = make_train_model(batch_size[3], NUM_NEURONS[1]) # Q1, batch size 32, 10 hidden neurons

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

