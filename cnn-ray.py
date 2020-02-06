# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import os
import ray
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ray.init()
# global prediction


def compute_accuracy(v_xs, v_ys):
    # global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def construct_network():
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

    ## conv1 layer ##
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # x_image: 28*28*1 output size: 28*28*32
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # output size: 14*14*32

    ## conv2 layer ##
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
                         b_conv2)  # 14*14*32 -> 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # 14*14*64 -> 7*7*64

    ## func1 layer ##
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## func2 layer ##

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return xs, ys, train_step, keep_prob

# define actor for structure
@ray.remote(num_gpus=1)
class NeuralNetOnGPU(object):
    def __init__(self, mnist_data):
        self.mnist = mnist_data
        # Set an environment variable to tell TensorFlow which GPUs to use. Note
        # that this must be done before the call to tf.Session.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in ray.get_gpu_ids()])
        with tf.Graph().as_default():
            with tf.device("/gpu:0"):
                self.xs, self.ys, self.train_step, self.keep_prob = construct_network()
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

    def train(self, num_steps):
        for i in range(num_steps):
            print(i)
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={
                        self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: 0.5})
            if i % 50 == 0:
                print("test")


# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

nn = NeuralNetOnGPU.remote(mnist)
start = time.time()
nn.train.remote(1000)
end = time.time()
print('execution time: ' + str(end-start) + 's')
ray.shutdown()
# accuracy = ray.get(nn.get_accuracy.remote())  # ray.get 从对象ID中进行数据的读取（python对象）
# print("Accuracy is {}.".format(accuracy))
# print(ray.get(nn.compute_accuracy.remote(mnist.test.images[:1000], mnist.test.labels[:1000])))
# ray.get 从对象ID中进行数据的读取（python对象）
# accuracy = ray.get(nn.compute_accuracy.remote(mnist.test.images[:1000], mnist.test.labels[:1000]))
# print("Accuracy is {}.".format(accuracy))
# start = time.time()
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={
#              xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
#     if i % 50 == 0:
#         print(compute_accuracy(
#             mnist.test.images[:1000], mnist.test.labels[:1000]))
# end = time.time()
# print('execution time: ' + str(end-start) + 's')
