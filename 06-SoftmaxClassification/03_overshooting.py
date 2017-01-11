import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder("float", [None, 3]) # 1(bias), x1, x2
Y = tf.placeholder("float", [None, 3]) # A, B, C

W = tf.Variable(tf.zeros([3, 3]))

# Our hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# Modify lerning rate
learning_rate = 10
# learning_rate = 0.0001

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# Before starting, initialize the variables.
init = tf.global_variables_initializer()

# Launch the graph.
with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print(a, sess.run(tf.arg_max(a, 1)))

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print(all, sess.run(tf.arg_max(all, 1)))