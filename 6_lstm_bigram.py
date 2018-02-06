
# coding: utf-8

# 

# Deep Learning
# =============
# 
# Assignment 6
# ------------
# 
# After training a skip-gram model in `5_word2vec.ipynb`, the goal of this notebook is to train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

# In[114]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve


# In[115]:

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# In[116]:

def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data
  
text = read_data(filename)
print('Data size %d' % len(text))


# Create a small validation set.

# In[117]:

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


# Utility functions to map characters to vocabulary IDs and back.

# In[118]:

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))

#DAF understanding
#print(first_letter)
#print(vocabulary_size)


# Function to generate a training batch for the LSTM model.

# Simple LSTM Model.

# ---
# Problem 1
# ---------
# 
# You might have noticed that the definition of the LSTM cell involves 4 matrix multiplications with the input, and 4 matrix multiplications with the output. Simplify the expression by using a single matrix multiply for each, and variables that are 4 times larger.
# 
# ---

# ---
# Problem 2
# ---------
# 
# We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters like 'a'. Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will lead to a very sparse representation that is very wasteful computationally.
# 
# a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs themselves.
# 
# b- Write a bigram-based LSTM, modeled on the character LSTM above.
# 
# c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to this [article](http://arxiv.org/abs/1409.2329).
# 
# ---

# In[126]:

batch_size=64
num_unrollings=10

class BigramBatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()


  def _next_batch(self):
    """Generate a single batch from the current bigram positions in the data. The bigrams are Idx (an embedding)"""
    batch = list()
    for b in range(self._batch_size):
      first_char = self._text[self._cursor[b]]

      if self._cursor[b] + 1 == self._text_size:
        second_char = ' '
      else:
        second_char = self._text[self._cursor[b] + 1]
      
      batch.append(char2id(first_char) * vocabulary_size + char2id(second_char))
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch


  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches


bigram_train_batches = BigramBatchGenerator(train_text, batch_size, num_unrollings)
bigram_valid_batches = BigramBatchGenerator(valid_text, 1, 1)
#DAF valid num_unrollings=1 una sola llamada a lstm_cell, y una sola actualizacion de state y output

def characters_from_embed(embeddings):
  r = [ '(' + id2char(e//vocabulary_size) + id2char(e%vocabulary_size) + ')' for e in embeddings]
  return r

def bigram_characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return ['({0},{1})'.format(id2char(c//vocabulary_size), id2char(c % vocabulary_size))
          for c in np.argmax(probabilities,1)]


def bigram_batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * len(batches[0])
  for b in batches:
    #print("b:",b)
    #print("bigram:", bigram_characters(b))
    s = [''.join(x) for x in zip(s, characters_from_embed(b))]
  return s

print(bigram_batches2string(bigram_train_batches.next()))
print(bigram_batches2string(bigram_train_batches.next()))
print(bigram_batches2string(bigram_valid_batches.next()))
print(bigram_batches2string(bigram_valid_batches.next()))


# In[127]:

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0] #labels.shape=(640,27)


def sample_distribution(distribution):  #normalized probabilities: que todas suman 1
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample_embedding(prediction): #parece que distibution esta en prediction[0], es la primera columna
  """Turn a (column) prediction into embed sample."""
  p = np.zeros(shape=[1,], dtype=np.int) 
  p[0] = sample_distribution(prediction[0])
  return p #p.shape=[1,]

def bigram_random_distribution(): #las normaliza en el return
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, embedding_size])
  return b/np.sum(b, 1)[:,None] #shape=(1,27)

#DAF understanding
#d= bigram_random_distribution()
#print(d)
#print(sum(d[0]))
#print(sample_embedding(d))


# In[128]:

num_nodes = 64
embedding_size = vocabulary_size * vocabulary_size

graph = tf.Graph()
with graph.as_default():
 

  # Parameters:
  # All the gates: input, previous output, and bias.
  cix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes * 4], -0.1, 0.1)) # cix.shape=[729, 256]
  cim = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], -0.1, 0.1)) # cim.shape=[64, 256]
  cib = tf.Variable(tf.zeros([1, num_nodes * 4])) # cib.shape=[1,256]
    
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False) # saved_output.shape=[64,64]
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False) # saved_state.shape=[64,64]

  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], -0.1, 0.1)) # w.shape=[64,729]
  b = tf.Variable(tf.zeros([embedding_size])) # b.shape =[729,]
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state, train=False): #i:input shape=[64,27], o:saved_output (shape=[64,64]), state:saved_state (shape=[64,64])
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""

    embed = tf.nn.embedding_lookup(cix ,i)   
    if train:
        embed = tf.nn.dropout(embed, 0.5)
    
    all_gates = embed + tf.matmul(o, cim) + cib # all_gates.shape=[64, 256]
    #  embedding_lookup(cix.shape=[729, 256], i.shape=[64,]) --> embedding_lookup.shape=[64, 256]
    #+ matmul(o.shape=[64,64], cim.shape=[64, 256]) --> matmul.shape=[64, 256], +.shape=[64, 256]
    #+ cib.shape=[1, 256] -->  all_gates.shape=[64, 256] 
       
    input_gate = tf.sigmoid(all_gates[:, 0:num_nodes])
    
    forget_gate = tf.sigmoid(all_gates[:, num_nodes:2*num_nodes])
    
    update = all_gates[:, 2*num_nodes:3*num_nodes]
    
    state = forget_gate * state + input_gate * tf.tanh(update)
    
    output_gate = tf.sigmoid(all_gates[:, 3*num_nodes:])
    
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append( 
      tf.placeholder(tf.int32, shape=[batch_size])) #<--- OJO: batch.shape=[64,] -> 64 embeddings = 64 ids ,

  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output # shape=[64,64]
  state = saved_state # shape=[64,64]
    
  for i in train_inputs: # 10 x shape=[64,]
    output, state = lstm_cell(i, output, state, train=True)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):

    # Classifier.
    
    logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b) # shape=[640, 729]
       
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.one_hot(tf.concat(train_labels,0), embedding_size)))    
    #labels.shape=[640, 729]
    
    #loss = tf.reduce_mean(
      #tf.nn.sparse_softmax_cross_entropy_with_logits(
        #logits,  tf.concat(0, train_labels)))
        
    #DAF understanding
    #print("logits: ", logits)
    #print("labels:", tf.one_hot(tf.concat(train_labels,0), embedding_size))
    #print("softmax_cross_entropy_with_logits:", tf.nn.softmax_cross_entropy_with_logits(
            #logits=logits, labels=tf.one_hot(tf.concat(train_labels,0), embedding_size)))
        
  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  gradients, v = zip(*optimizer.compute_gradients(loss)) 
  #This is the first part of minimize(). 
  #It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable". 

  #DAF: entremedias se hace el "Gradient Clipping"  
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)

  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step) 
  #This is the second part of minimize(). It returns an Operation that applies gradients.
  #grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients()


  # Predictions.
  train_prediction = tf.nn.softmax(logits) # shape=[640, 729]
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1]) #un embeding shape=[1]
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes])) #shape=[1,64]
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes])) #shape=[1,64]
  
  reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])), 
                                saved_sample_state.assign(tf.zeros([1, num_nodes])))

  sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)


  with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]):        
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b)) #1 bigram 1hot encoding: shape=[1,729]


# In[129]:

#num_steps = 7001
num_steps = 1101
summary_frequency = 100

with tf.Session(graph=graph) as session:
    
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0

  for step in range(num_steps):

    batches = bigram_train_batches.next()
    feed_dict = dict()   
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i] 
    
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    
    if step % summary_frequency == 0:
        
      if step > 0: 
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
    
      labels = np.concatenate(list(batches)[1:]) #labels.shape=[640,]
      # convert to one-hot-encodings
      noembed_labels = np.zeros(predictions.shape) #predictions.shape=[640, 729]
      for i, j in enumerate(labels): #DAF i=0,1,2, j=el valor
        noembed_labels[i, j] = 1.0

      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, noembed_labels))))
    
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample_embedding(bigram_random_distribution()) #feed es una bigram sampleado [1,729]. El primero de cada seq al azar
          sentence = characters_from_embed(feed)[0]
          reset_sample_state.run()
          
          for _ in range(79):
            #DAF feed alimenta, via la var sample input, el proceso de sample_prediction
            #y luego se vuelve a samplear un nuevo feed (una nueva letra), basado en la prediccion de sample_prediction 
            prediction = sample_prediction.eval({sample_input: feed}) 
            feed = sample_embedding(prediction)
            sentence += characters_from_embed(feed)[0]
            
          print(sentence)  
        print('=' * 80)
        
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
    
      for _ in range(valid_size): #DAF: valid_size = 1000, valid_batches = BatchGenerator(valid_text, 1, 1)
        b = bigram_valid_batches.next() #b.shape=[1,1+1=2]
        predictions = sample_prediction.eval({sample_input: b[0]}) #se hace la prediccion con el primer bigram
        labels = np.zeros((1, embedding_size))
        labels[0, b[1]] = 1.0
        valid_logprob = valid_logprob + logprob(predictions, labels) # la label es el segundo bigram
  
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))


# ---
# Problem 3
# ---------
# 
# (difficult!)
# 
# Write a sequence-to-sequence LSTM which mirrors all the words in a sentence. For example, if your input is:
# 
#     the quick brown fox
#     
# the model should attempt to output:
# 
#     eht kciuq nworb xof
#     
# Refer to the lecture on how to put together a sequence-to-sequence model, as well as [this article](http://arxiv.org/abs/1409.3215) for best practices.
# 
# ---
