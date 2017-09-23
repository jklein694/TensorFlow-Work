import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_Data', one_hot=True)

# reset everything to rerun in jupyter
tf.reset_default_graph()

# config
batch_size = 100
learning_rate = 0.5
training_epochs = 5
logs_path = "logs"

# Build out all of your place holders and variables for model structure

# These can be whatever we want to set them to
num_nodes_hl1 = 500
num_nodes_hl2 = 500
num_nodes_hl3 = 500

# Number of output classes equals 10
# Our classes are also in one hot encoding

n_classes = 10


# Set the height and wide of model
# These are x and y variables and they must be one dimensional each
# So the first dimension always equals None and the second must equal the dimension of your X and y variable
# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder('float', shape=[None, 784], name='x_placeholder')
    # target 10 output classes
    y = tf.placeholder('float', shape=[None, n_classes], name='y_placeholder')


# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W1 = tf.Variable(tf.random_normal([784, num_nodes_hl1], name='hidden_layer_1'))
    W2 = tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2], name='hidden_layer_2'))
    W3 = tf.Variable(tf.random_normal([num_nodes_hl2, num_nodes_hl3], name='hidden_layer_3'))
    out_w = tf.Variable(tf.random_normal([num_nodes_hl3, n_classes], name='hidden_layer_1'))


# bias
with tf.name_scope("biases"):
    b1 = tf.Variable(tf.random_normal([num_nodes_hl1]))
    b2 = tf.Variable(tf.random_normal([num_nodes_hl2]))
    b3 = tf.Variable(tf.random_normal([num_nodes_hl3]))
    out_b = tf.Variable(tf.random_normal([n_classes]))

# implement model
with tf.name_scope("softmax"):
    # y is our prediction
    l1 = tf.add(tf.matmul(x, W1), b1)

    # Add activation function to run this data
    l1 = tf.nn.relu(l1, name='relu_l1')

    l2 = tf.add(tf.matmul(l1, W2), b2)
    l2 = tf.nn.relu(l2, name='relu_l2')

    l3 = tf.add(tf.matmul(l2, W3), b3)
    l3 = tf.nn.relu(l2, name='relu_l3')

    output = tf.add(tf.matmul(l3, out_w), out_b)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cost = tf.reduce_mean(-(tf.reduce_sum(y * tf.log(output), reduction_indices=[1])))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    # Our stochastic gradient descent function to determine what our function should do next to decrease cost
    optimizer = tf.train.AdamOptimizer().minimize(cost, name='optimizer')

with tf.name_scope('Accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a summary for our cost and accuracy
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.initialize_all_variables())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # perform training cycles
    for epoch in range(training_epochs):

        # number of batches in one epoch
        batch_count = int(mnist.train.num_examples / batch_size)

        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # perform the operations we defined earlier on batch
            _, summary = sess.run([optimizer, summary_op], feed_dict={x: batch_x, y: batch_y})

            # write log
            writer.add_summary(summary, epoch * batch_count + i)

        if epoch % 5 == 0:
            print
            "Epoch: ", epoch
    print
    "Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print
    "done"


