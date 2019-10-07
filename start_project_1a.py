#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
# import pylab as plt
import matplotlib.pyplot as plt
import time
import sys

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# Build the graph for the deep net
def l3_ffn(x, neuron_size, weight_decay_beta):
    """Feedforward net with 1 hidden layer
    """
    sum_regularization = 0
    with tf.name_scope('hidden'):
        weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, neuron_size], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
        biases  = tf.Variable(tf.zeros([neuron_size]), name='biases')
        h  = tf.nn.relu(tf.matmul(x, weights) + biases)
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    with tf.name_scope('linear'):
        weights = tf.Variable(tf.truncated_normal([neuron_size, NUM_CLASSES], stddev=1.0/math.sqrt(float(neuron_size))), name='weights')
        biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        u = tf.matmul(h, weights) + biases
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    
    return u, sum_regularization

def l4_ffn(x, neuron_size, weight_decay_beta):
    """Feedforward net with 2 hidden layer.
    """
    sum_regularization = 0
    with tf.name_scope('hidden'):
        weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, neuron_size], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
        biases  = tf.Variable(tf.zeros([neuron_size]), name='biases')
        h  = tf.nn.relu(tf.matmul(x, weights) + biases)
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([neuron_size, neuron_size], stddev=1.0/math.sqrt(float(neuron_size))), name='weights')
        biases  = tf.Variable(tf.zeros([neuron_size]), name='biases')
        h  = tf.nn.relu(tf.matmul(x, weights) + biases)
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    with tf.name_scope('linear'):
        weights = tf.Variable(tf.truncated_normal([neuron_size, NUM_CLASSES], stddev=1.0/math.sqrt(float(neuron_size))), name='weights')
        biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        u = tf.matmul(h, weights) + biases
        sum_regularization += weight_decay_beta * tf.nn.l2_loss(weights)
    
    return u, sum_regularization

def make_train_model(train_x, test_x, train_y, test_y, batch_sizes: list, neuron_sizes: list, weight_decay_beta: list, layers=3):
    train_acc = {}
    test_acc = {}
    time_elapsed = {}
    for weight_decay in weight_decay_beta:
        train_acc[weight_decay] = {}
        test_acc[weight_decay] = {}
        time_elapsed[weight_decay] = {}
        for neuron_size in neuron_sizes:
            # Create the model
            print(f'Training {neuron_size} for hidden layer')
            x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
            y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

            if layers==3:
                logits, regularizer = l3_ffn(x, neuron_size, weight_decay)
            elif layers == 4:
                logits, regularizer = l4_ffn(x, neuron_size, weight_decay)
            else:
                print('Invalid number of layers given.')
                return

            cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
            loss = tf.reduce_mean(cost + regularizer)

            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)

            correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

            for batch_size in batch_sizes:
                train_acc[weight_decay][neuron_size] = {}
                test_acc[weight_decay][neuron_size] = {}
                time_elapsed[weight_decay][neuron_size] = {}
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    train_acc[weight_decay][neuron_size][batch_size] = []
                    test_acc[weight_decay][neuron_size][batch_size] = []
                    time_elapsed[weight_decay][neuron_size][batch_size] = []
                    for fold in range(num_folds):
                        print(f'Fold number: {fold+1}')
                        n = int(train_x.shape[0] / num_folds)
                        fold_start, fold_end = fold*n, (fold+1)*n
                        x_test, y_test = test_x[fold_start:fold_end], test_y[fold_start:fold_end]
                        x_train  = np.append(train_x[:fold_start], train_x[fold_end:], axis=0)
                        y_train = np.append(train_y[:fold_start], train_y[fold_end:], axis=0) 
                        # print(test_index)

                        for i in range(epochs):
                            # Time
                            start_time = time.time()
                            # Handle in batches
                            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
                                train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
                            train_acc[weight_decay][neuron_size][batch_size].append(accuracy.eval(feed_dict={x: x_train, y_: y_train}))
                            test_acc[weight_decay][neuron_size][batch_size].append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
                            time_elapsed[weight_decay][neuron_size][batch_size].append(time.time() - start_time)
                            if i % 100 == 0:
                                print('iter %d: accuracy %g'%(i, train_acc[weight_decay][neuron_size][batch_size][i]))
    return train_acc, test_acc, time_elapsed

def plot_acc(filename: str, train_acc: dict, test_acc: dict, weight_decay_betas: list, neuron_size: list, batch_size: list):
    fig, ax = plt.subplots()
    for weight_decay in weight_decay_betas:
        for neuron in neuron_size:
            for batch in batch_size:
                ax.plot(range(epochs), train_acc[weight_decay][neuron][batch], linewidth=2, label=f'Train accuracy, neuron: {neuron}, batch_size: {batch}')
                ax.plot(range(epochs), test_acc[weight_decay][neuron][batch], linewidth=2, label=f'Test accuracy, neuron: {neuron}, batch_size: {batch}')
    ax.legend(loc='lower right')
    plt.savefig(filename)
    plt.show()

def plot_time(filename: str, time_elapsed: dict, weight_decay_beta:list, neuron_size: list, batch_size: list):
    fig, ax = plt.subplots()
    for weight_decay in weight_decay_beta:
        for neuron in neuron_size:
            for batch in batch_size:
                ax.plot(range(epochs), time_elapsed[weight_decay][neuron][batch], linewidth=2, label=f'Batch_size: {batch} time elapsed(s)')
    ax.legend(loc='lower right')
    plt.savefig(filename)
    plt.show()

def plot_3l_4l_comp(filename: str, train_acc_3: dict, test_acc_3: dict, train_acc_4: dict, test_acc_4: dict, weight_decay: list, neuron: list, batch: list):
    fig, ax = plt.subplots()
    ax.plot(range(epochs), train_acc_3[weight_decay[0]][neuron[0]][batch[0]], linewidth=2, label='Train accuracy, 3 layer network')
    ax.plot(range(epochs), test_acc_3[weight_decay[0]][neuron[0]][batch[0]], linewidth=2, label='Test accuracy, test 3 layer network')
    ax.plot(range(epochs), train_acc_4[weight_decay[1]][neuron[1]][batch[1]], linewidth=2, label='Train accuracy, train 4 layer network')
    ax.plot(range(epochs), test_acc_4[weight_decay[1]][neuron[1]][batch[1]], linewidth=2, label='Test accuracy, test 4 layer network')
    ax.legend(loc='lower right')
    plt.savefig(filename)
    plt.show()

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

NUM_FEATURES = 21
NUM_CLASSES = 3

num_folds = 5
learning_rate = 0.01
epochs = 5000
weight_decay_betas = [0, float('10e-3'), float('10e-6'), float('10e-9'), float('10e-12')]
batch_size = [4, 8, 16, 32, 64]
neuron_size = [5,10,15,20,25]
seed = 10
test_split = 0.3
np.random.seed(seed)

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 # one hot matrix

# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

test_split_num = int(len(trainX) * test_split)
train_x, test_x = trainX[:test_split_num], trainX[test_split_num:]
train_y, test_y = trainY[:test_split_num], trainY[test_split_num:]

# Q1, batch size 32, 10 hidden neurons
train_acc, test_acc, _ = make_train_model(train_x, test_x, train_y, test_y, [batch_size[3]], [neuron_size[1]], [weight_decay_betas[2]])
plot_acc('plots/part1_Q1a.png', train_acc, test_acc, [weight_decay_betas[2]], [neuron_size[1]], [batch_size[3]])

# Q2, batch sizes {4, 8, 16, 32, 64}, 10 hidden neurons
train_acc, test_acc, time_elapsed = make_train_model(train_x, test_x, train_y, test_y, batch_size, [neuron_size[1]], [weight_decay_betas[2]])
plot_time('plots/part1_Q2a.png', time_elapsed, [weight_decay_betas[2]], [neuron_size[1]], batch_size)
plot_acc('plots/part1_Q2c.png', train_acc, test_acc, [weight_decay_betas[2]], [neuron_size[1]], batch_size)

optimal_batch_size = batch_size[4]
plot_acc('plots/part1_Q2c_optimal.png', train_acc, test_acc, [weight_decay_betas[2]], [neuron_size[1]], [optimal_batch_size])

# # Q3, batch size ?, {5,10,15,20,25} hidden neurons
# train_acc, test_acc,_ = make_train_model(train_x, test_x, train_y, test_y, [optimal_batch_size], neuron_size, [weight_decay_betas[2]])
# plot_acc('plots/part1_Q3a.png', train_acc, test_acc, [weight_decay_betas[2]], neuron_size, [optimal_batch_size])

# optimal_neuron_size = neuron_size[4]
# plot_acc('plots/part1_Q3c.png', train_acc, test_acc, [weight_decay_betas[2]], [optimal_neuron_size], [optimal_batch_size])

# # Q4, batch size ?, ? hidden neurons, decay parameters {0, 10e−3, 10e−6, 10e−9, 10e−12}
# train_acc, test_acc,_ = make_train_model(train_x, test_x, train_y, test_y, [optimal_batch_size], [optimal_neuron_size], [weight_decay_betas[2]])
# plot_acc('plots/part1_Q4a.png', train_acc, test_acc, weight_decay_betas, [optimal_neuron_size], [optimal_batch_size])

# optimal_weight_decay = weight_decay_betas[3]
# plot_acc('plots/part1_Q4c.png', train_acc, test_acc, [optimal_weight_decay], [optimal_neuron_size], [optimal_batch_size])

# # Q5, 3 or 4 layers
# # Run with Q4 code
# train_acc_2, test_acc_2,_ = make_train_model(train_x, test_x, train_y, test_y, [batch_size[3]], [neuron_size[1]], [weight_decay_betas[2]], layers=4)
# plot_acc('plots/part1_Q5a.png', train_acc_2, test_acc_2, [batch_size[3]], [neuron_size[1]], [weight_decay_betas[2]])
# plot_3l_4l_comp('plots/part1_Q5b.png', train_acc, test_acc, train_acc_2, test_acc_2, [optimal_batch_size, batch_size[3]], [optimal_neuron_size, neuron_size[1]], [optimal_weight_decay, weight_decay_betas[2]])
