# DAF: An approach to solve Problem 4.2
# "Try to get the best performance you can using a convolutional net. Look for example at the classic LeNet5 architecture, adding 
# Dropout, and/or adding learning rate decay."
# HERE we use Dropout, AvgPooling, *optional* L2 Regularization & LearningRate decay


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
patch_size = 5
num_channels = 1 # grayscale


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
      predictions.shape[0])

# Create image size function based on input, filter size, padding and stride
# 2 convolutions only with 2 pooling
def output_size_pool(input_size, conv_filter_size, pool_filter_size, padding, conv_stride, pool_stride):
    if padding == 'same':
        padding = -1.00
    elif padding == 'valid':
        padding = 0.00
    else:
        return None
    # After convolution 1
    output_1 = (((input_size - conv_filter_size - 2*padding) / conv_stride) + 1.00)
    # After pool 1
    output_2 = (((output_1 - pool_filter_size - 2*padding) / pool_stride) + 1.00)    
    # After convolution 2
    output_3 = (((output_2 - conv_filter_size - 2*padding) / conv_stride) + 1.00)
    # After pool 2
    output_4 = (((output_3 - pool_filter_size - 2*padding) / pool_stride) + 1.00)  
    return int(output_4)

final_image_size = output_size_pool(input_size=image_size, conv_filter_size=5, pool_filter_size=2, padding='valid', conv_stride=1, pool_stride=2)
print(final_image_size)


batch_size = 16
# Depth is the number of output channels 
# On the other hand, num_channels is the number of input channels set at 1 previously
depth = 32
num_hidden = 64
beta = 0.001
eval_batch_size = 64


graph = tf.Graph()

with graph.as_default():

  '''Input data'''
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  #DAF
  #tf_valid_dataset = tf.constant(valid_dataset)
  #tf_test_dataset = tf.constant(test_dataset)
  eval_data = tf.placeholder( tf.float32, shape=(eval_batch_size, image_size, image_size, num_channels))

  '''Variables'''
  # Convolution 1 Layer
  # Input channels: num_channels = 1
  # Output channels: depth = 16
  layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
    
  # Convolution 2 Layer
  # Input channels: depth = 16
  # Output channels: depth = 16
  layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    
  # First Fully Connected Layer (Densely Connected Layer)
  # Use neurons to allow processing of entire image
  final_image_size = output_size_pool(input_size=image_size, conv_filter_size=5, pool_filter_size=2, padding='valid', conv_stride=1, pool_stride=2)
  layer3_weights = tf.Variable(tf.truncated_normal([final_image_size * final_image_size * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
  # Second Fully Connected Layer
  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
  # Readout layer: Softmax Layer
  layer5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  '''Model'''
  def model(data):
    # First Convolutional Layer with Pooling
    conv_1 = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='VALID')
    hidden_1 = tf.nn.relu(conv_1 + layer1_biases)
    pool_1 = tf.nn.avg_pool(hidden_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        
    # Second Convolutional Layer with Pooling
    conv_2 = tf.nn.conv2d(pool_1, layer2_weights, strides=[1, 1, 1, 1], padding='VALID')
    hidden_2 = tf.nn.relu(conv_2 + layer2_biases)
    pool_2 = tf.nn.avg_pool(hidden_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        
    # First Fully Connected Layer
    shape = pool_2.get_shape().as_list()
    reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    keep_prob = 0.5
    hidden_drop = tf.nn.dropout(hidden, keep_prob)
        
    # Second Fully Connected Layer
    hidden_2 = tf.nn.relu(tf.matmul(hidden_drop, layer4_weights) + layer4_biases)
    hidden_2_drop = tf.nn.dropout(hidden_2, keep_prob)        
    # Readout Layer: Softmax Layer
    return tf.matmul(hidden_2_drop, layer5_weights) + layer5_biases

  '''Training computation'''
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  # Loss function with L2 Regularization 
  # regularizers = tf.nn.l2_loss(layer4_weights) + \
                 # tf.nn.l2_loss(layer5_weights)
  # loss = tf.reduce_mean(loss + beta * regularizers)

  '''Optimizer'''
  # Decaying learning rate
  global_step = tf.Variable(0)  # count the number of steps taken.
  start_learning_rate = 0.05
  learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  '''Predictions for the training, validation, and test data'''
  train_prediction = tf.nn.softmax(logits)
  #DAF
  #valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  #test_prediction = tf.nn.softmax(model(tf_test_dataset))

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < eval_batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, num_labels), dtype=np.float32)
    for begin in xrange(0, size, eval_batch_size):
      end = begin + eval_batch_size
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-eval_batch_size:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


num_steps = 30000
with tf.Session(graph=graph) as session:

  tf.global_variables_initializer().run()
  print('Initialized')
  
  for step in range(num_steps):
    
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    
    if (step % 100 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      #print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
      print('Validation error: %.1f%%' % accuracy(eval_in_batches(valid_dataset, session), valid_labels))

      print(" -sleeping 20 seg, we do this so as not to burn the cpu-")
      time.sleep(20)
    
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  print('Test error: %.1f%%' % accuracy(eval_in_batches(test_dataset, session), test_labels))


