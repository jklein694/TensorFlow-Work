import tensorflow as tf
import numpy as np
from create_sentiment_features import create_features_set_and_labels


train_x, train_y, test_x, test_y = create_features_set_and_labels('pos.txt', 'neg.txt')
sess=tf.Session()
#First let's load meta graph and restore weights
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model-1000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
# This will print 2, which is the value of bias that weTensorFlow vs Keras/model.ckpt.data-00000-of-00001 saved
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    print(sess.run('W1:0'))



#Now, access the op that you want to run.


print(sess.run())