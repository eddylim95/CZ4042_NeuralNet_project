# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Assignment_2'))
	print(os.getcwd())
except:
	pass

import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm_notebook as tqdm
import sys

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 200
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_

def cnn(images, num_filter_1, num_filter_2, dropout=False):
    # NHWC format
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    #Conv 1, 50 filters of window size 9x9, VALID padding, and ReLU
    with tf.variable_scope('CNN_Layer1'):
        conv1 = tf.layers.conv2d(
            images,
            filters=num_filter_1,
            kernel_size=[9,9],
            padding='VALID',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=2,
            strides=2,
            padding='VALID')
        if dropout:
            pool1 = tf.layers.dropout(pool1, 0.25)

    with tf.variable_scope('Char_CNN_Layer2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=num_filter_2,
            kernel_size=[5,5],
            padding='VALID',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=2,
            strides=2,
            padding='VALID')
        if dropout:
            pool2 = tf.layers.dropout(pool2, 0.25)

    # pool2_ = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])
    dim = pool2.get_shape()[1].value * pool2.get_shape()[2].value * pool2.get_shape()[3].value 
    pool2_ = tf.reshape(pool2, [-1, dim])
    
    # Fully connected layer size 300
    f3 = tf.layers.dense(pool2_, 300, activation=tf.nn.relu)
    if dropout:
        f3 = tf.layers.dropout(f3, 0.5)

    #Softmax, size 10. Note that softmax happens at softmax_entropy step
    f4 = tf.layers.dense(f3, NUM_CLASSES, activation=None)

    return conv1, pool1, conv2, pool2, f4

def plot_acc(filename: str, epochs: int, test_acc: list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    ax.plot(range(epochs), test_acc, label=f'Test accuracy')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_all_acc(filename: str, epochs: int, acc_list: list, layer_1_search: list, layer_2_search: list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    idx = 0
    for i in layer_1_search:
        for j in layer_2_search:
            ax.plot(range(epochs), acc_list[idx], label=f'Test accuracy for layer1={i}, layer2={j}')
            idx += 1
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_cost(filename: str, epochs: int, train_cost: list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    ax.plot(range(epochs), train_cost, label=f'Train cost')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_optimizer_cost(filename: str, epochs: int, cost_list: list, optimizer_list: list, dropout_list: list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    for i, optimizer in enumerate(optimizer_list):
        text = '' if dropout_list[i] else 'out'
        ax.plot(range(epochs), cost_list[i], label=f'Train cost for {optimizer} with{text} dropout')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_optimizer_acc(filename: str, epochs: int, acc_list: list, optimizer_list: list, dropout_list: list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    for i, optimizer in enumerate(optimizer_list):
        text = '' if dropout_list[i] else 'out'
        ax.plot(range(epochs), acc_list[i], label=f'Test accuracy for {optimizer} with{text} dropout')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_layer(filename, layer, num_filters):
    """Note that there is an assumption that num_filters is a multiple of 10
    """
    plt.figure()
    plt.gray()
    layer_ = np.array(layer)
    for i in range(num_filters):
        plt.subplot(num_filters/10, 10, i+1)
        plt.axis('off')
        plt.imshow(layer_[0,:,:,i])
    plt.savefig(filename)

def plot_feature_maps(X, conv_1, pool_1, conv_2, pool_2, i):
    # Test pattern
    plt.figure()
    plt.gray()
    X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(X_show)
    plt.savefig(f'partA_plots/q1a_test_pattern_{i}.png')

    # Conv 1
    plot_layer(f'partA_plots/q1a_conv1_{i}.png' ,conv_1, 50)

    # Pool 1
    plot_layer(f'partA_plots/q1a_pool1_{i}.png' ,pool_1, 50)

    # Conv 2
    plot_layer(f'partA_plots/q1a_conv2_{i}.png' ,conv_2, 60)

    # Pool 2
    plot_layer(f'partA_plots/q1a_pool2_{i}.png' ,pool_2, 60)


# %%
trainX, trainY = load_data('data_batch_1')
print(trainX.shape, trainY.shape)

testX, testY = load_data('test_batch_trim')
print(testX.shape, testY.shape)

trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

# Check GPU status
print('Gpu status: ' + str(tf.test.is_gpu_available()))


# %%
# Q1
# Create the model
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

conv_1, pool_1, conv_2, pool_2, logits = cnn(x, 50, 60)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

N = len(trainX)
idx = np.arange(N)

train_cost = []
test_acc = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in tqdm(range(epochs)):
        np.random.shuffle(idx)
        train_X, train_Y = trainX[idx], trainY[idx]
        train_cost_ = []
        # Handle in batches
        for start, end in zip(range(0, len(train_X), batch_size), range(batch_size, len(train_X), batch_size)):
            _, batch_cost = sess.run([train_step, loss], {x: train_X[start:end], y_: train_Y[start:end]})
            train_cost_.append(batch_cost)
        _, batch_cost = sess.run([train_step, loss], {x: train_X[end:], y_: train_Y[end:]})
        train_cost_.append(batch_cost)
        train_cost.append(np.mean(np.array(train_cost_), axis=0))
        test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
        if e % 10 == 0:
            print('epoch', e, 'accuracy', test_acc[e])

    # Plot feature maps
    for i in range(2):
        ind = np.random.randint(low=0, high=10000)
        X = trainX[ind,:]
        conv_1_, pool_1_, conv_2_, pool_2_ = sess.run([conv_1, pool_1, conv_2, pool_2],
                                                    {x: X.reshape(1,IMG_SIZE*IMG_SIZE*NUM_CHANNELS)})
        plot_feature_maps(X, conv_1_, pool_1_, conv_2_, pool_2_, i)

plot_cost('partA_plots/q1a_1.png', epochs, train_cost)
plot_acc('partA_plots/q1a_2.png', epochs, test_acc)


# %%
# Q2
layer_1_search = [4, 8, 16, 32, 64]
layer_2_search = [4, 8, 16, 32, 64]

all_acc = []
for i in layer_1_search:
    for j in layer_2_search:
        # Create the model
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

        conv_1, pool_1, conv_2, pool_2, logits = cnn(x, i, j)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        N = len(trainX)
        idx = np.arange(N)

        test_acc = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for e in tqdm(range(epochs)):
                np.random.shuffle(idx)
                train_X, train_Y = trainX[idx], trainY[idx]
                # Handle in batches
                for start, end in zip(range(0, len(train_X), batch_size), range(batch_size, len(train_X), batch_size)):
                    sess.run([train_step, loss], {x: train_X[start:end], y_: train_Y[start:end]})
                test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
                if e % 10 == 0:
                    print('epoch', e, 'accuracy', test_acc[e])
        all_acc.append(test_acc)


# %%
for i, size in enumerate(layer_1_search):
    plot_all_acc(f'partA_plots/q2_{i}.png', epochs, all_acc[i*5:i*5+5], [size], layer_2_search)


# %%
# Q3

comparisons = [
               {'optimizer': 'gd', 'dropout': False},
               {'optimizer': 'momentum', 'dropout': False},
               {'optimizer': 'rms', 'dropout': False},
               {'optimizer': 'adam', 'dropout': False},
               {'optimizer': 'gd', 'dropout': True}
]

all_cost = []
all_acc = []
for item in comparisons:
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    conv_1, pool_1, conv_2, pool_2, logits = cnn(x, 32, 64, dropout=item['dropout'])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    if item['optimizer'] == 'gd':
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif item['optimizer'] == 'momentum':
        train_step = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    elif item['optimizer'] == 'rms':
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif item['optimizer'] == 'adam':
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)

    train_cost = []
    test_acc = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in tqdm(range(epochs)):
            np.random.shuffle(idx)
            train_X, train_Y = trainX[idx], trainY[idx]
            # Handle in batches
            train_cost_ = []
            for start, end in zip(range(0, len(train_X), batch_size), range(batch_size, len(train_X), batch_size)):
                _, batch_cost = sess.run([train_step, loss], {x: train_X[start:end], y_: train_Y[start:end]})
                train_cost_.append(batch_cost)
            train_cost.append(np.mean(np.array(train_cost_), axis=0))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            if e % 10 == 0:
                print('epoch', e, 'accuracy', test_acc[e])
    all_cost.append(train_cost)
    all_acc.append(test_acc)


# %%
optimizer_list = [item['optimizer'] for item in comparisons]
dropout_list = [item['dropout'] for item in comparisons]

plot_optimizer_cost(f'partA_plots/q3_1.png', epochs, all_cost, optimizer_list, dropout_list)
plot_optimizer_acc(f'partA_plots/q3_2.png', epochs, all_acc, optimizer_list, dropout_list)

