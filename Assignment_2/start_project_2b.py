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

import numpy as np
import pandas
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import time
from tqdm import tqdm_notebook as tqdm

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

batch_size = 128
no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def cnn_model(x, model_type, dropout=False):
    if model_type == 'char':
        input_layer = tf.reshape(
            tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

        # if dropout:
        #     input_layer = tf.layers.dropout(input_layer, 0.1)

        with tf.variable_scope('Char_CNN_Layer1'):
            conv1 = tf.layers.conv2d(
                input_layer,
                filters=N_FILTERS,
                kernel_size=FILTER_SHAPE1,
                padding='VALID',
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=POOLING_WINDOW,
                strides=POOLING_STRIDE,
                padding='SAME')
            if dropout:
                pool1 = tf.layers.dropout(pool1, 0.25)
        with tf.variable_scope('Char_CNN_Layer2'):
            conv2 = tf.layers.conv2d(
                pool1,
                filters=N_FILTERS,
                kernel_size=FILTER_SHAPE2,
                padding='VALID',
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(
                conv2,
                pool_size=POOLING_WINDOW,
                strides=POOLING_STRIDE,
                padding='SAME')
            if dropout:
                pool2 = tf.layers.dropout(pool2, 0.25)

        pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

        logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

        # if dropout:
        #     logits = tf.layers.dropout(logits, 0.5)

        return input_layer, logits
    
    elif model_type == 'word':
        word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
        
        word_list = tf.unstack(word_vectors, axis=1)

        input_layer = tf.reshape(
            word_vectors, [-1, MAX_DOCUMENT_LENGTH, HIDDEN_SIZE, 1])
        # if dropout:
        #     input_layer = tf.layers.dropout(input_layer, 0.1)

        with tf.variable_scope('Word_CNN_Layer1'):
            conv1 = tf.layers.conv2d(
                input_layer,
                filters=N_FILTERS,
                kernel_size=[20, 20],
                padding='VALID',
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=POOLING_WINDOW,
                strides=POOLING_STRIDE,
                padding='SAME')
            if dropout:
                pool1 = tf.layers.dropout(pool1, 0.25)
        with tf.variable_scope('Word_CNN_Layer2'):
            conv2 = tf.layers.conv2d(
                pool1,
                filters=N_FILTERS,
                kernel_size=FILTER_SHAPE2,
                padding='VALID',
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(
                conv2,
                pool_size=POOLING_WINDOW,
                strides=POOLING_STRIDE,
                padding='SAME')
            if dropout:
                pool2 = tf.layers.dropout(pool2, 0.25)

        pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

        logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

        # if dropout:
        #     logits = tf.layers.dropout(logits, 0.5)

        return input_layer, logits
    else:
        raise Exception(f'No model type named {model_type}')

def rnn_model(x, model_type, cell_type='GRU', num_layers=1, dropout=False):
    if model_type == 'char':
        input_layer = tf.one_hot(x, 256)
        input_layer = tf.unstack(input_layer, axis=1)

        cells_ = []
        for i in range(num_layers):
            # Define cell type
            if cell_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
            elif cell_type == 'VANILLA':
                cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
            elif cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
            else:
                raise Exception(f'No cell type matches {cell_type}')

            # Define dropout
            if dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)

            cells_.append(cell)

        # Multi-layer rnn cells
        cells = tf.nn.rnn_cell.MultiRNNCell(cells_)
        _, encoding = tf.nn.static_rnn(cells, input_layer, dtype=tf.float32)

        encoding = np.array(encoding).flatten()

        # Dense layer
        logits = tf.layers.dense(encoding[-1], MAX_LABEL, activation=None)
        # if dropout:
        #     logits = tf.layers.dropout(logits, 0.5)

        return input_layer, logits

    elif model_type == 'word':
        word_vectors = tf.contrib.layers.embed_sequence(
            x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

        word_list = tf.unstack(word_vectors, axis=1)

        cells_ = []
        for i in range(num_layers):
            # Define cell type
            if cell_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
            elif cell_type == 'VANILLA':
                cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
            elif cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
            else:
                raise Exception(f'No cell type matches {cell_type}')

            # Define dropout
            if dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)

            cells_.append(cell)

        # Multi-layer rnn cells
        cells = tf.nn.rnn_cell.MultiRNNCell(cells_)
        _, encoding = tf.nn.static_rnn(cells, word_list, dtype=tf.float32)

        encoding = np.array(encoding).flatten()

        # Dense layer
        logits = tf.layers.dense(encoding[-1], MAX_LABEL, activation=None)

        # if dropout:
        #     logits = tf.layers.dropout(logits, 0.5)

        return word_list, logits
    else:
        raise Exception(f'No model type named {model_type}')

def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open('test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)

    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values

    return x_train, y_train, x_test, y_test

def data_read_words():
    x_train, y_train, x_test, y_test = [], [], [], []
    
    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))
    
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values
    
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)

    return x_train, y_train, x_test, y_test, no_words

def plot_acc(filename: str, epochs: int, test_acc: list, dropout_acc=[]):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    ax.plot(range(epochs), test_acc, label=f'Test accuracy')
    if dropout_acc != []:
        ax.plot(range(epochs), dropout_acc, label=f'Test accuracy with dropout')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_cost(filename: str, epochs: int, train_cost: list, dropout_cost=[]):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    ax.plot(range(epochs), train_cost, label=f'Train cost')
    if dropout_cost != []:
        ax.plot(range(epochs), dropout_cost, label=f'Train cost with dropout')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_all_acc(filename: str, epochs: int, accuracies: list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    labels = ['GRU', 'RNN', 'LSTM', '2-layer GRU', 'GRU with gradient clipping']
    for i,accuracy in enumerate(accuracies):
        ax.plot(range(epochs), accuracy, label=f'Test accuracy for {labels[i]}')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def plot_all_cost(filename: str, epochs: int, costs: list):
    fig, ax = plt.subplots(figsize=[12.8,9.6])
    labels = ['GRU', 'RNN', 'LSTM', '2-layer GRU', 'GRU with gradient clipping']
    for i,cost in enumerate(costs):
        ax.plot(range(epochs), cost, label=f'Train cost for {labels[i]}')
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.show()


# %%
# Q1
x_train, y_train, x_test, y_test = read_data_chars()

print(np.array(x_train).shape)
print(np.array(x_test).shape)

tf.reset_default_graph()

# Create the model
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

inputs, logits = cnn_model(x, model_type='char')

# Optimizer
entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss = []
test_acc = []
time_taken = 0
# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            _, batch_cost = sess.run([train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss.append(np.mean(np.array(loss_), axis=0))
        test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('iter: %d, entropy: %g'%(e, loss[e]))
    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q1_1.png', no_epochs, loss)
plot_acc('partB_plots/q1_2.png', no_epochs, test_acc)

# With dropout
# Create the model
tf.reset_default_graph()
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

inputs, logits = cnn_model(x, model_type='char', dropout=True)

# Optimizer
entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss_dropout = []
test_acc_dropout = []
time_taken = 0
# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            _, batch_cost = sess.run([train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss_dropout.append(np.mean(np.array(loss_), axis=0))
        test_acc_dropout.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('iter: %d, entropy: %g'%(e, loss_dropout[e]))
    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q1_3.png', no_epochs, loss, loss_dropout)
plot_acc('partB_plots/q1_4.png', no_epochs, test_acc, test_acc_dropout)


# %%
# Q2
global n_words

x_train, y_train, x_test, y_test, n_words = data_read_words()

tf.reset_default_graph()

# Create the model
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

word_list, logits = cnn_model(x, model_type='word')

entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss = []
test_acc = []
time_taken = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    # training
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            word_list_, _, batch_cost = sess.run([word_list, train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss.append(np.mean(np.array(loss_), axis=0))
        test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('epoch: %d, entropy: %g'%(e, loss[e]))

    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q2_1.png', no_epochs, loss)
plot_acc('partB_plots/q2_2.png', no_epochs, test_acc)

# With dropout
# Create the model
tf.reset_default_graph()
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

word_list, logits = cnn_model(x, model_type='word', dropout=True)

entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss_dropout = []
test_acc_dropout = []
time_taken = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    # training
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            word_list_, _, batch_cost = sess.run([word_list, train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss_dropout.append(np.mean(np.array(loss_), axis=0))
        test_acc_dropout.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('epoch: %d, entropy: %g'%(e, loss_dropout[e]))

    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q2_3.png', no_epochs, loss, loss_dropout)
plot_acc('partB_plots/q2_4.png', no_epochs, test_acc, test_acc_dropout)


# %%
# Q3
x_train, y_train, x_test, y_test = read_data_chars()

print(np.array(x_train).shape)
print(np.array(x_test).shape)

tf.reset_default_graph()

# Create the model
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

inputs, logits = rnn_model(x, model_type='char')

entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss = []
test_acc = []
time_taken = 0
# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    # training
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            _, batch_cost = sess.run([train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss.append(np.mean(np.array(loss_), axis=0))
        test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('iter: %d, entropy: %g'%(e, loss[e]))
    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q3_1.png', no_epochs, loss)
plot_acc('partB_plots/q3_2.png', no_epochs, test_acc)

# With dropout
# Create the model
tf.reset_default_graph()
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

inputs, logits = rnn_model(x, model_type='char', dropout=True)

entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss_dropout = []
test_acc_dropout = []
time_taken = 0
# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    # training
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            _, batch_cost = sess.run([train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss_dropout.append(np.mean(np.array(loss_), axis=0))
        test_acc_dropout.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('iter: %d, entropy: %g'%(e, loss_dropout[e]))
    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q3_3.png', no_epochs, loss, loss_dropout)
plot_acc('partB_plots/q3_4.png', no_epochs, test_acc, test_acc_dropout)


# %%
# Q4
global n_words

x_train, y_train, x_test, y_test, n_words = data_read_words()

tf.reset_default_graph()

# Create the model
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

word_list, logits = rnn_model(x, model_type='word')

entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss = []
test_acc = []
time_taken = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    # training
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            word_list_, _, batch_cost = sess.run([word_list, train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss.append(np.mean(np.array(loss_), axis=0))
        test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('epoch: %d, entropy: %g'%(e, loss[e]))

    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q4_1.png', no_epochs, loss)
plot_acc('partB_plots/q4_2.png', no_epochs, test_acc)

# With dropout
# Create the model
tf.reset_default_graph()
x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)

word_list, logits = rnn_model(x, model_type='word', dropout=True)

entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

loss_dropout = []
test_acc_dropout = []
time_taken = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Time
    start_time = time.time()
    # training
    for e in tqdm(range(no_epochs)):
        loss_ = []
        # Handle in batches
        for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
            word_list_, _, batch_cost = sess.run([word_list, train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
            loss_.append(batch_cost)
        loss_dropout.append(np.mean(np.array(loss_), axis=0))
        test_acc_dropout.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

        if e%10 == 0:
            print('epoch: %d, entropy: %g'%(e, loss_dropout[e]))

    time_taken = time.time() - start_time

print(time_taken)
plot_cost('partB_plots/q4_3.png', no_epochs, loss, loss_dropout)
plot_acc('partB_plots/q4_4.png', no_epochs, test_acc, test_acc_dropout)


# %%
# Q6

comparisons = [
               {'cell_type': 'GRU', 'num_layers': 1, 'grad_clip': False},
               {'cell_type': 'VANILLA', 'num_layers': 1, 'grad_clip': False},
               {'cell_type': 'LSTM', 'num_layers': 1, 'grad_clip': False},
               {'cell_type': 'GRU', 'num_layers': 2, 'grad_clip': False},
               {'cell_type': 'GRU', 'num_layers': 1, 'grad_clip': True}
               ]

# for char model
x_train, y_train, x_test, y_test = read_data_chars()

print(np.array(x_train).shape)
print(np.array(x_test).shape)

losses = []
accuracies = []
for i,item in enumerate(comparisons):
    tf.reset_default_graph()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    inputs, logits = rnn_model(x, model_type='char', cell_type=item['cell_type'], num_layers=item['num_layers'])

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

    # Gradient clipping
    if item['grad_clip']:
        minimizer = tf.train.AdamOptimizer()
        grads_and_vars = minimizer.compute_gradients(entropy)
        grad_clipping = tf.constant(2.0, name="grad_clipping")
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
            clipped_grads_and_vars.append((clipped_grad, var))
        train_op = minimizer.apply_gradients(clipped_grads_and_vars)
    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    loss = []
    test_acc = []
    time_taken = 0
    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Time
        start_time = time.time()
        for e in tqdm(range(no_epochs)):
            loss_ = []
            # Handle in batches
            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
                _, batch_cost = sess.run([train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
                loss_.append(batch_cost)
            loss.append(np.mean(np.array(loss_), axis=0))
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            if e%10 == 0:
                print('iter: %d, entropy: %g'%(e, loss[e]))
    losses.append(loss)
    accuracies.append(test_acc)


# %%
plot_all_cost('partB_plots/q6_1.png', no_epochs, losses)
plot_all_acc('partB_plots/q6_2.png', no_epochs, accuracies)


# %%
# Q7
global n_words

x_train, y_train, x_test, y_test, n_words = data_read_words()

word_list, logits = cnn_model(x, model_type='word')
comparisons = [
               {'cell_type': 'GRU', 'num_layers': 1, 'grad_clip': False},
               {'cell_type': 'VANILLA', 'num_layers': 1, 'grad_clip': False},
               {'cell_type': 'LSTM', 'num_layers': 1, 'grad_clip': False},
               {'cell_type': 'GRU', 'num_layers': 2, 'grad_clip': False},
               {'cell_type': 'GRU', 'num_layers': 1, 'grad_clip': True}
               ]

losses_words = []
accuracies_words = []
for i,item in enumerate(comparisons):
    tf.reset_default_graph()
    
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    word_list, logits = rnn_model(x, model_type='char', cell_type=item['cell_type'], num_layers=item['num_layers'])

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

    # Gradient clipping
    if item['grad_clip']:
        minimizer = tf.train.AdamOptimizer()
        grads_and_vars = minimizer.compute_gradients(entropy)
        grad_clipping = tf.constant(2.0, name="grad_clipping")
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
            clipped_grads_and_vars.append((clipped_grad, var))
        train_op = minimizer.apply_gradients(clipped_grads_and_vars)
    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    loss = []
    test_acc = []
    time_taken = 0
    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Time
        start_time = time.time()
        for e in tqdm(range(no_epochs)):
            loss_ = []
            # Handle in batches
            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train), batch_size)):
                word_list_, _, batch_cost = sess.run([word_list, train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})
                loss_.append(batch_cost)
            loss.append(np.mean(np.array(loss_), axis=0))
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            if e%10 == 0:
                print('iter: %d, entropy: %g'%(e, loss[e]))
    losses_words.append(loss)
    accuracies_words.append(test_acc)


# %%
plot_all_cost('partB_plots/q6_3.png', no_epochs, losses)
plot_all_acc('partB_plots/q6_4.png', no_epochs, accuracies)

