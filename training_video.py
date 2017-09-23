import tensorflow as tf

import numpy as np
from create_sentiment_features import create_features_set_and_labels

train_x, train_y, test_x, test_y = create_features_set_and_labels('pos.txt', 'neg.txt')

# Build out all of your place holders and variables for model structure

# These can be whatever we want to set them to
num_nodes_hl1 = 500
num_nodes_hl2 = 500
num_nodes_hl3 = 500

# Number of output classes equals 10
# Our classes are also in one hot encoding
n_classes = len(train_y[0])
input_dim = len(train_x[0])

# We can set our batch size to whatever we can fit into memory
batch_size = 100

# Set the height and wide of model
# These are x and y variables and they must be one dimensional each
# So the first dimension always equals None and the second must equal the dimension of your X and y variable
x = tf.placeholder('float', shape=[None, input_dim], name='x_placeholder')
y = tf.placeholder('float', shape=[None, n_classes], name='y_placeholder')

def neural_network_model(data):
    ''' Model or NN Structure '''

    # Initialize hidden layer's variables (Inputs, Weights, and Biases) and take random sample from batch training data

    # Shape equals input data, and number of nodes
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([input_dim, num_nodes_hl1], name='hidden_layer_1')),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl1]))}

    # The output shape of hidden_layer_1 is the num_nodes_hl1, so that is the input for the next layer
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2], name='hidden_layer_1')),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl3]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl2, num_nodes_hl3], name='hidden_layer_1')),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl3]))}

    # Lastly we need the output layer, which must have the same output shape as our number of classes or our y shape
    output_layer = {'weights': tf.Variable(tf.random_normal([num_nodes_hl3, n_classes], name='hidden_layer_1')),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Build models function for ** (input * weights) + biases **
    # input = data
    # weights = hidden_layer_n['weights']
    # biases = hidden_layer_n['biases']

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])

    # Add activation function to run this data
    l1 = tf.nn.relu(l1, name='relu_l1')

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2, name='relu_l2')

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3, name='relu_l3')
    print(l3, output_layer['weights'], output_layer['biases'])

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

# At this point we have not trained any data, we just built the model or instructions.
# We need to separate the data into batches and loop over our model for our epochs.

def train_neural_network(x):
    '''Train our data through our model'''

    prediction = neural_network_model(x)

    # Cost function to determine how wrong our prediction was
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y, name='cost'))

    # Our stochastic gradient descent function to determine what our function should do next to decrease cost
    optimizer = tf.train.AdamOptimizer().minimize(cost, name='optimizer')

    # Determine how many epochs we would like to use
    hm_epoch = 10

    # Initialize all global variables
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    # saver = tf.train.Saver([neural_network_model(x)])

    # Activate our sessions and repeat over for loop
    with tf.Session() as sess:

        # compile all the tf assigned variables (tf.Variables)
        sess.run(init_op)

        # Loop over the defined number of epochs
        for epoch in range(hm_epoch):
            # initialize loss at 0
            epoch_loss = 0

            i = 0
            batch_count = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                batch_count += 1

                # we use _ because it returns a tuple where the first value is always none, so we dont care about it
                _, c = sess.run([optimizer, cost], feed_dict= {x: batch_x, y: batch_y})

                # add loss to epoch loss
                epoch_loss += c
                i += batch_size

            print('Epoch ', epoch, ' completed out of ', hm_epoch, ', loss: ', epoch_loss)

        # Determine the correct amount of classifications
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy ', accuracy.eval({x:test_x, y: test_y}))

        # Save the variables to disk.
        # save_path = saver.save(sess, "my_test_model", global_step=10)
        # print("Model saved in file: %s" % save_path)

# Run all functions
train_neural_network(x)
