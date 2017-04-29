import numpy as np
import tensorflow as tf

def W_var(shape, stddev=0.001, name="W"):
  initial = tf.random_normal(shape, stddev=stddev)
  var = tf.Variable(initial, name=name)
  return var

def b_var(length, name="b"):
  initial = tf.constant(0.0, shape=[length])
  return tf.Variable(initial, name=name)

