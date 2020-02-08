import os
import ray
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data

ray.init(num_gpus=8)
# consruct neural network


def construct_network():
    # [None, 784]: data structure. total number of attribute is 28*28=784 with uncertain row number (batch size, can be of any size.)
    x = tf.placeholder(tf.float32, [None, 784])
    # total number of attribute is 10 (0-9) with uncertain row number
    y_ = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))  # Weights
    b = tf.Variable(tf.zeros([10]))  # biase
    y = tf.nn.softmax(tf.matmul(x, W) + b) # y = wx + b
    # y_: real，y: prediction
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                  tf.log(y), reduction_indices=[1]))  # loss function
    # Use gradientdescentoptimizer to min the loss function
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 1:search by row. tf.equal: 对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast： convert correct_prediction to float32

    return x, y_, train_step, accuracy

# define actor for structure
@ray.remote(num_gpus=1)  # actor gpu数量为1
class NeuralNetOnGPU(object):
    def __init__(self, mnist_data):
        self.mnist = mnist_data
        # Set an environment variable to tell TensorFlow which GPUs to use. Note
        # that this must be done before the call to tf.Session.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in ray.get_gpu_ids()])
        with tf.Graph().as_default():
            with tf.device("/gpu:0"):
                self.x, self.y_, self.train_step, self.accuracy = construct_network()
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
            # load dataset by batch
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            # train
            self.sess.run(self.train_step, feed_dict={
                          self.x: batch_xs, self.y_: batch_ys})
            if (i% 50):
                print(self.get_accuracy())
    def get_accuracy(self):
        return self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images,
                                                       self.y_: self.mnist.test.labels})


# load MNIST dataset，并告诉Ray如何序列化定制类。
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Create the actor. 实例化actor并运行构造函数
nn = NeuralNetOnGPU.remote(mnist)

# Run a few steps of training and print the accuracy.
nn.train.remote(200)
accuracy = ray.get(nn.get_accuracy.remote()) # ray.get 从对象ID中进行数据的读取（python对象）
print("Accuracy is {}.".format(accuracy))
