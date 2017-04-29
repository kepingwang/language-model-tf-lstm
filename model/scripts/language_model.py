import numpy as np
import tensorflow as tf
import scripts.net_helper as net_helper

class LanguageModel(object):

  def __init__(self, batch_size, T, global_step, vocab_size):

    init_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.96
    
    clip_norm = 5
    hidden_size = 128 # hidden vector size (assume shared by all)
    num_layers = 3

    predict_steps = 20

    input_data = tf.placeholder(tf.int64, [batch_size, T], name="inputData")
    self._input_data = input_data
    labels = tf.placeholder(tf.int64, [batch_size, T], name="labels")
    self._labels = labels
    
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            hidden_size, forget_bias=0.0, state_is_tuple=True)

    attn_cell = lstm_cell
    # TODO: dropout and batchnorm
    # if is_training and config.keep_prob < 1:
    #   def attn_cell():
    #     return tf.contrib.rnn.DropoutWrapper(
    #         lstm_cell(), output_keep_prob=config.keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(num_layers)], state_is_tuple=True)

    embeddings = net_helper.W_var([vocab_size, hidden_size], name="embeddings")
    self._embeddings = embeddings
    inputs = tf.nn.embedding_lookup(embeddings, input_data)

    # TODO: dropout and batchnorm
    # if is_training and config.keep_prob < 1:
    #   inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    state = cell.zero_state(batch_size, dtype=tf.float32)
    with tf.variable_scope("RNN"):
      for t in range(T):
        if t > 0: tf.get_variable_scope().reuse_variables()
        cell_output, state = cell(inputs[:, t, :], state)
        outputs.append(cell_output)

    # reshaped output 
    output = tf.reshape(tf.stack(outputs, axis=1), [-1, hidden_size])

    softmax_w = net_helper.W_var([hidden_size, vocab_size], name="project_out_W")
    softmax_b = net_helper.b_var(vocab_size, name="project_out_b")
    logits = tf.matmul(output, softmax_w) + softmax_b

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=tf.reshape(labels, [-1])
    )
    loss = tf.reduce_mean(loss)
    self._loss = loss
    self._perplexity = tf.exp(loss)

    # ==== Define train_op ====
    self._lr = tf.train.exponential_decay(
        init_learning_rate, global_step, 1000, 0.96)

    optimizer = tf.train.AdamOptimizer(self._lr)
    grads, tvars = zip(*optimizer.compute_gradients(loss))
    grads, _ = tf.clip_by_global_norm(grads, clip_norm) # gradient clippingx

    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # ==== Predict Output after Input ====
    # previously already have outputs and state
    with tf.variable_scope("RNN", reuse=True):
      for t in range(predict_steps):
        cell_output, state = cell(outputs[-1], state)
        outputs.append(cell_output)

    prediction_output = tf.reshape(tf.stack(outputs, axis=1), [-1, hidden_size])
    prediction_logits = prediction_output @ softmax_w + softmax_b
    predictions = tf.reshape(
      tf.argmax(prediction_logits, axis=1),
      [batch_size, T+predict_steps])
    self._predictions = predictions

  @property
  def input_data(self):
    """Placeholder to feed data in"""
    return self._input_data

  @property
  def labels(self):
    """Placeholder to feed data in"""
    return self._labels

  @property
  def predictions(self):
    return self._predictions

  @property
  def embeddings(self):
    return self._embeddings

  @property
  def loss(self):
    return self._loss

  @property
  def perplexity(self):
    return self._perplexity

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
