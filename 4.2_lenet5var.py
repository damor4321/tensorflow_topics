# DAF: First approach to Lenet5arch from http://deeplearning.net/tutorial/lenet.html
# HERE we use MaxPooling

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



#image_size = 28
#num_labels = 10
#num_channels = 1 # grayscale
batch_size = 16

c1_depth = 6
c1_ker_sz = 5
c3_depth = 16
c3_ker_sz = 6
c5_depth = 120
c5_ker_sz = 6
num_hidden = 84

eval_batch_size = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)) #shape=(16,28,28,1)
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels)) #shape=(16,10)

  eval_data = tf.placeholder( tf.float32, shape=(eval_batch_size, image_size, image_size, num_channels))

  #DAF: we comment the 2 following lines,  We comment the following 2 lines, applying the strategy of not making the prediction from the whole valid_dataset or from the whole test_dataset. But only a prediction based on minibatch of the dataset size "eval_batch_size"
  #tf_valid_dataset = tf.constant(valid_dataset)
  #tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  c1_weights = tf.Variable(tf.truncated_normal(
    [c1_ker_sz, c1_ker_sz, num_channels, c1_depth], stddev=0.1)) #shape=(5,5,1,6)
  c1_biases = tf.Variable(tf.zeros([c1_depth])) #shape=(6,)

  c3_weights = tf.Variable(tf.truncated_normal(
    [c3_ker_sz, c3_ker_sz, c1_depth, c3_depth], stddev=0.1)) #shape=(6,6,6,16)
  c3_biases = tf.Variable(tf.constant(1.0, shape=[c3_depth])) #shape=(16,)

  c5_weights = tf.Variable(tf.truncated_normal(
    [c5_ker_sz, c5_ker_sz, c3_depth, c5_depth], stddev=0.1)) #shape=(6,6,16,120)
  c5_biases = tf.Variable(tf.constant(1.0, shape=[c5_depth])) #shape=(120,)
  c5_conv_dim = (((((image_size+1)//2) + 1) // 2) + 1 )//2 #4

  fc_weights = tf.Variable(tf.truncated_normal(
    [c5_conv_dim * c5_conv_dim * c5_depth, num_hidden], stddev=0.1)) #shape=(4*4*120,64)=(1920,64)
  fc_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden])) #shape=(64,)

  out_weights = tf.Variable(tf.truncated_normal(
    [num_hidden, num_labels], stddev=0.1)) #shape=(64,10)
  out_biases = tf.Variable(tf.constant(1.0, shape=[num_labels])) #shape=(10,)  
  
  # Model.
  def model(data):

      print(data.get_shape().as_list())
    
      conv = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
      hidden = tf.nn.relu(conv + c1_biases)
      print(conv.get_shape().as_list())

      pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      print(pooled.get_shape().as_list())
    
      conv = tf.nn.conv2d(pooled, c3_weights, [1, 1, 1, 1], padding='SAME')
      hidden = tf.nn.relu(conv + c3_biases)
      pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      shape = pooled.get_shape().as_list()
      print(shape)
    
      conv = tf.nn.conv2d(pooled, c5_weights, [1, 1, 1, 1], padding='SAME')
      hidden = tf.nn.relu(conv + c5_biases)
      pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      shape = pooled.get_shape().as_list()
      print(shape)
    
      reshape = tf.reshape(pooled, [shape[0], shape[1] * shape[2] * shape[3]])
      hidden = tf.nn.relu(tf.matmul(reshape, fc_weights) + fc_biases)

      return tf.matmul(hidden, out_weights) + out_biases


  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  optimizer = tf.train.AdagradOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)

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

  #DAF: Comentamos las 2 lineas siguiente, siguiendo la estratagema de no hacer la prediccion a partir de todo el valid_dataset ni todo el test_dataset
  #Sino solo una estimacion de prediccion basada en minibatch de estos datasets de tamanyo eval_batch_size

  #valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  #test_prediction = tf.nn.softmax(model(tf_test_dataset))


#import sys
#sys.exit()

num_steps = 3001

with tf.Session(graph=graph) as session:

  tf.global_variables_initializer().run()
  print('Initialized')
  
  for step in range(num_steps):
    
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      #print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
      print('Validation error: %.1f%%' % accuracy(eval_in_batches(valid_dataset, session), valid_labels))
    
    if(step % 100 == 0):
      print(" -sleeping 15 seg, we do this so as not to burn the cpu-")
      time.sleep(15)
    
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  print('Test error: %.1f%%' % accuracy(eval_in_batches(test_dataset, session), test_labels))


