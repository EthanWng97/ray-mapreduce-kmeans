from __future__ import division, print_function, absolute_import
import os
import ray
import numpy as np  # linear algebra
import h5py
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting

# Hyper Parameter
batch_size = 86
epochs = 20

ray.init(num_gpus=8, include_webui=True, use_pickle=False)
# Translate data to color


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:, :-1]


def translate(x):
    xx = np.ndarray((x.shape[0], 4096, 3))
    for i in range(x.shape[0]):
        xx[i] = array_to_color(x[i])
    # Free Memory
    del x

    return xx

with h5py.File("/Users/wangyifan/Google Drive/3dmnist/full_dataset_vectors.h5", 'r') as h5:
    X_train, y_train = h5["X_train"][:], h5["y_train"][:]
    X_test, y_test = h5["X_test"][:], h5["y_test"][:]

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

X_train = translate(X_train).reshape(-1, 16, 16, 16, 3)
X_test = translate(X_test).reshape(-1, 16, 16, 16, 3)

def construct_network():

    with tf.name_scope('inputs'):
        x_input = tf.placeholder(tf.float32, shape=[None, 16, 16, 16, 3])
        y_input = tf.placeholder(tf.float32, shape=[None, 10])
        keep_rate = tf.placeholder(tf.float32)

    with tf.name_scope("layer_a"):
        # conv => 16*16*16 feature: 16
        conv1 = tf.layers.conv3d(inputs=x_input, filters=16, kernel_size=[
                                 3, 3, 3], padding='same', activation=tf.nn.relu)
        # conv => 16*16*16 feature: 32
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[
                                 3, 3, 3], padding='same', activation=tf.nn.relu)
        # pool => 8*8*8
        pool3 = tf.layers.max_pooling3d(
            inputs=conv2, pool_size=[2, 2, 2], strides=2)

    with tf.name_scope("layer_c"):
        # conv => 8*8*8 feature: 64
        conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[
                                 3, 3, 3], padding='same', activation=tf.nn.relu)
        # conv => 8*8*8 feature: 128
        conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[
                                 3, 3, 3], padding='same', activation=tf.nn.relu)
        # pool => 4*4*4
        pool6 = tf.layers.max_pooling3d(
            inputs=conv5, pool_size=[2, 2, 2], strides=2)

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=True)

    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 4*4*4*128])
        dense = tf.layers.dense(
            inputs=flattening, units=1024, activation=tf.nn.relu)
        # (1-keep_rate) is the probability that the node will be kept
        dropout = tf.layers.dropout(
            inputs=dense, rate=keep_rate, training=True)

    with tf.name_scope("prediction"):
        prediction = tf.layers.dense(inputs=dropout, units=10)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=y_input))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    return x_input, y_input, optimizer, keep_rate, cost, accuracy


@ray.remote(num_gpus=1)
class CNN_ON_RAY(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in ray.get_gpu_ids()])
        with tf.Graph().as_default():
            with tf.device("/gpu:0"):
                self.x_input, self.y_input, self.optimizer, self.keep_rate, self.cost, self.accuracy = construct_network()
                # Allow this to run on CPUs if there aren't any GPUs.
                config = tf.ConfigProto(allow_soft_placement=True)
                #### normal network
                # init = tf.initialize_all_variables()
                # sess = tf.Session()
                # sess.run(init)
                ####
                self.sess = tf.Session(config=config)
                # Initialize the network.
                init = tf.global_variables_initializer()
                self.sess.run(init)

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
        iterations = int(len(X_train)/batch_size) + 1
        # run epochs
        for epoch in range(epochs):
            print('Epoch', epoch, 'started', end='')
            epoch_loss = 0
            # mini batch
            for itr in range(iterations):
                mini_batch_x = X_train[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y = y_train[itr*batch_size: (itr+1)*batch_size]
                _optimizer, _cost = self.sess.run([self.optimizer, self.cost], feed_dict={
                    self.x_input: mini_batch_x, self.y_input: mini_batch_y, self.keep_rate: 0.7})
                epoch_loss += _cost

            #  using mini batch in case not enough memory
            acc = 0
            itrs = int(len(X_test)/batch_size) + 1
            for itr in range(itrs):
                mini_batch_x_test = X_test[itr *
                                                batch_size: (itr+1)*batch_size]
                mini_batch_y_test = y_test[itr *
                                                batch_size: (itr+1)*batch_size]
                acc += self.sess.run(self.accuracy, feed_dict={
                    self.x_input: mini_batch_x_test, self.y_input: mini_batch_y_test, self.keep_rate: 0.7})

            print(' Testing Set Accuracy:', acc/itrs)#, ' Time elapse: ',


start = time.time()
nn = CNN_ON_RAY.remote()
train_id = nn.train.remote(X_train[:1000], y_train[:1000], X_test[:100],
                     y_test[:100], epochs=10, batch_size=32)
ray.get(train_id)
end = time.time()
print('execution time: ' + str(end-start) + 's')
