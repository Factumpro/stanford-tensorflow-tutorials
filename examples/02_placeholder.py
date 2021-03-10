""" Placeholder and feed_dict example
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 02
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# For new version of tensorflow, which has eager execution enabled by default
tf.compat.v1.disable_eager_execution()

# Example 1: feed_dict with placeholder

# Implement tf.compat.v1.

# a is a placeholderfor a vector of 3 elements, type tf.float32
a = tf.compat.v1.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)


# use the placeholder as you would a constant
c = a + b  # short for tf.add(a, b)

writer = tf.compat.v1.summary.FileWriter('graphs/placeholders', tf.compat.v1.get_default_graph())
with tf.compat.v1.Session() as sess:
    # compute the value of c given the value of a is [1, 2, 3]
    print(sess.run(c, {a: [1, 2, 3]}))                 # [6. 7. 8.]
writer.close()


# Example 2: feed_dict with variables
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.compat.v1.Session() as sess:
    print(sess.run(b))                                 # >> 21
    # compute the value of b given the value of a is 15
    print(sess.run(b, feed_dict={a: 15}))              # >> 45
