import tensorflow as tf
import numpy as np
from create_sentiment_features import create_features_set_and_labels


train_x, train_y, test_x, test_y = create_features_set_and_labels('pos.txt', 'neg.txt')
train_x
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

n_classes = len(train_y[0])
input_dim = len(train_x[0])
print(input_dim)
print(n_classes)


# Set the height and wide of model
# These are x and y variables and they must be one dimensional each
# So the first dimension always equals None and the second must equal the dimension of your X and y variable
# input images
with tf.name_scope('input'):
    # None -> batch size can be any size training set feature length
    x = tf.placeholder('float', shape=[None, input_dim], name='x_placeholder')
    # target 10 output classes
    y = tf.placeholder('float', shape=[None, n_classes], name='y_placeholder')


# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W1 = tf.Variable(tf.random_normal([input_dim, num_nodes_hl1], name='W1'))
    W2 = tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2], name='W2'))
    W3 = tf.Variable(tf.random_normal([num_nodes_hl2, num_nodes_hl3], name='W3'))
    out_w = tf.Variable(tf.random_normal([num_nodes_hl3, n_classes],  name='out_w'))
    tf.add_to_collection('vars', W1)
    tf.add_to_collection('vars', W2)
    tf.add_to_collection('vars', W3)
    tf.add_to_collection('vars', out_w)

# bias
with tf.name_scope("biases"):
    b1 = tf.Variable(tf.random_normal([num_nodes_hl1], name='b1'))
    b2 = tf.Variable(tf.random_normal([num_nodes_hl2], name='b2'))
    b3 = tf.Variable(tf.random_normal([num_nodes_hl3], name='b3'))
    out_b = tf.Variable(tf.random_normal([n_classes], name='out_b'))
    tf.add_to_collection('vars', b1)
    tf.add_to_collection('vars', b2)
    tf.add_to_collection('vars', b3)
    tf.add_to_collection('vars', out_b)

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

    output = tf.add(tf.matmul(l3, out_w), out_b, name='output')

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


saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'model')
# create a summary for our cost and accuracy
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session
summary_op = tf.summary.merge_all()

saver.save(sess, 'model')

# saves a model every 2 hours and maximum 4 latest models are saved.
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

# saver = tf.train.Saver([W1, W2, W3, out_w, b1, b2, b3, out_b])


with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.initialize_all_variables())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver.save(sess, 'model', global_step=1000)

    # perform training cycles
    for epoch in range(training_epochs):
        i = 0
        batch_count = 0

        if epoch != 2:
            saver.restore(sess, 'model-1000')

        while i < len(train_x):
            start = i
            end = i + batch_size

            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            batch_count += 1


            # perform the operations we defined earlier on batch
            _, summary = sess.run([optimizer, summary_op], feed_dict={x: batch_x, y: batch_y})

            # write log
            writer.add_summary(summary, epoch * batch_count + i)
            i += batch_size

        if epoch % 2 == 0:
            print("Epoch: ", epoch)
        saver.save(sess, 'model')
        tf.train.export_meta_graph('model_1')


    print("Accuracy: ", accuracy.eval(feed_dict={x: test_x, y: test_y}))
    print("done")


