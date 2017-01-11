import tensorflow as tf
import input_data


X = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
Y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
learning_rate = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

training_epochs = 25
display_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})

            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
