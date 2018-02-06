
# coding: utf-8

# Deep Learning
# =============
# Assignment 6, Problem 3
# ------------

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


# Utility functions to map characters to vocabulary IDs and back.
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
print(first_letter)
print(vocabulary_size)


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
import sys
sys.path.append(tf.__path__[0]+'/contrib')
#print(tf.__path__[0])
#print(sys.path)

#DAF: hemos copiado seq2seq_model.py y data_utils.py en el directorio de trabajo desde /home/david/coursera/models/tutorials/rnn/translate
import seq2seq_model as seq2seq_model


text = "the quick brown fox jumps over the lazy dog is an english sentence that can be translated to the following french one le vif renard brun saute par dessus le chien paresseux here is an extremely long french word anticonstitutionnellement"

def longest_word_size(text):
    return max(map(len, text.split()))


word_size = longest_word_size(text)
#print(word_size)



import string
num_nodes = 64
batch_size = 10

def create_model():
     return seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_size, #27 letras
                                   target_vocab_size=vocabulary_size, #27 letras
                                   buckets=[(word_size + 1, word_size + 2)], # only 1 bucket #DAF: buckets=[(25+1,25+2)] 
                                   size=num_nodes, # 64 nodos
                                   num_layers=3, #valor heredado de tutorial
                                   max_gradient_norm=5.0, #valor heredado de tutorial
                                   batch_size=batch_size, # 10 items
                                   learning_rate=0.5, #valor heredado de tutorial
                                   learning_rate_decay_factor=0.99, #valor heredado de tutorial
                                   use_lstm=True,
                                   forward_only=False) #valor heredado de tutorial


def get_batch():
    encoder_inputs = [np.random.randint(1, vocabulary_size, word_size + 1) for _ in xrange(batch_size)]
    decoder_inputs = [np.zeros(word_size + 2, dtype=np.int32) for _ in xrange(batch_size)]
    weights = [np.ones(word_size + 2, dtype=np.float32) for _ in xrange(batch_size)]
    for i in xrange(batch_size):
        r = random.randint(1, word_size)
        # leave at least a 0 at the end
        encoder_inputs[i][r:] = 0
        # one 0 at the beginning of the reversed word, one 0 at the end
        decoder_inputs[i][1:r+1] = encoder_inputs[i][:r][::-1]
        weights[i][r+1:] = 0.0
    return np.transpose(encoder_inputs), np.transpose(decoder_inputs), np.transpose(weights)


def strip_zeros(word):
    # 0 is the code for space in char2id()
    return word.strip(' ') #quita los espacios al principio y final de word


def evaluate_model(model, sess, words, encoder_inputs):
        
    correct = 0
    
    #DAF: CLAVE: MIENTRAS EN EL TRAINING decoder_inputs y weigths se pasan a model.step() con valores "coherentes" 
    # generados por "Ad-Hoc para el training" por getBatch()) 
    # Aqui el model.step esta forward_only=True. O sea: se usa para ESTIMAR LOS OUTPUTS. Por eso AL INICIO DEL BUCLE 
    # decoder_inputs y target_weights se pasan model.step VACIOS (a cero). Luego se van actualizando a cada step del bucle
    # en funcion de los OUTPUTS (output_logits) y volviendose a pasar (ya con contenido)
    
    # Finalmente despues del bucle, el decoder_inputs resultante producir la EVALUACION: comparandolo con las palabras
    # originales del batch (range_words)
    
    decoder_inputs = np.zeros((word_size + 2, batch_size), dtype=np.int32) #shape=[27,10], de 0s
    target_weights = np.zeros((word_size + 2, batch_size), dtype=np.float32) #shape=[27,10], de 0.0s
    target_weights[0,:] = 1.0 #target_weights[0] se pone todo a 1.0s
    
    is_finished = np.full(batch_size, False, dtype=np.bool_) #[False False False False False False False False False False]
    
    for i in xrange(word_size + 1): #DAF: caracter a carcacter de la WORD=25+1=26 (nada que ver con vocabulary_size=27)
        
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id=0, forward_only=True)
               
        # DAF: output_logits[0] is the predicted probability distribution for the first character. (EN LAS 10 PALABRAS)
        # I then run step() again in forward-only mode to get the predicted probability distribution for 
        # the second character, etc, etc.
        # output_logits[0].shape : (10, 27) *****este 27 es por vocabulary_size****, NO por (max_)word_size 
        # Y len(output_logits) = 27. Puesto que es una python list es xrange(word_size+1) = 0-26 = 27 
        # ***Este 27 es distinto al otro. Es por el tamaño word_size=25
        
        p = np.argmax(output_logits[i], axis=1) 
        
        #DAF: para este los caracteres ith de encoder_input tenemos su output_logits[i]
        #output_logits[i].shape : (filas=10=words, columnas=27=chars)
        #p son los 10 indices de la columna (de 0 a 26)(por axis=1) que tiene el mayor valor
        #print(p)
        
        #DAF: Actualizacion de decoder_inputs y target_weights en funcion de p (que es un resumen de output_logits[i])
        # de (el estado) de is_finished. #TODO: ENTENDER EL DETALLE DE ESTA ACTUALIZACION
        
        is_finished = np.logical_or(is_finished, p == 0) 
        
        decoder_inputs[i,:] = (1 - is_finished) * p 
        
        target_weights[i,:] = (1.0 - is_finished) * 1.0 
        
        
        #if np.all(is_finished):
            #break

    #DAF understanding
    #print("encoder_inputs: ", encoder_inputs, encoder_inputs.shape)
    print("decoder_inputs: \n",decoder_inputs, decoder_inputs.shape)    
    #print("target_weights: ", target_weights, target_weights.shape)
    #print("len(output_logits) :",len(output_logits)) # len(output_logits) =27
    #print("output_logits[0].shape :",output_logits[0].shape)
    #print("output_logits[0]:",output_logits[0]) #salida superverbosa
    print("P es ", np.argmax(output_logits[0], axis=1), np.argmax(output_logits[0], axis=1).shape)
    
    #DAF: EVALUACION: comparacion directa de output_word (generada a partir de decoder_inputs) O SEA LA PALABRA ESTIMADA, 
    #contra la reversed_word (inversa de la word original del batch) O SEA LA LABEL
    #si son iguales -> correct++
    
    for idx, l in enumerate(np.transpose(decoder_inputs)):
        
        reversed_word = ''.join(reversed(words[idx]))
        
        output_word = strip_zeros(''.join(id2char(i) for i in l))
        
        print(words[idx], '(reversed: {0})'.format(reversed_word),
              '-> [', output_word, '] ({0})'.format('OK' if reversed_word == output_word else 'KO'))
        
        
        if reversed_word == output_word:
            correct += 1
    
    return correct



def get_validation_batch(words):
    encoder_inputs = [np.zeros(word_size + 1, dtype=np.int32) for _ in xrange(batch_size)]
    for i, word in enumerate(words):
        for j, c in enumerate(word):
            encoder_inputs[i][j] = char2id(c)
    return np.transpose(encoder_inputs)



def validate_model(text, model, sess):
    words = text.split()
    nb_words = (len(words) / batch_size) * batch_size
    
    correct = 0
    for i in xrange(nb_words / batch_size):
        range_words = words[i * batch_size:(i + 1) * batch_size]
        
        encoder_inputs = get_validation_batch(range_words)
        
        correct += evaluate_model(model, sess, range_words, encoder_inputs)
    
    print('* correct: {0}/{1} -> {2}%'.format(correct, nb_words, (float(correct) / nb_words) * 100))
    print()


def reverse_text(nb_steps):
    with tf.Session() as session:
        model = create_model()
        tf.global_variables_initializer().run()

        for step in xrange(nb_steps):
            enc_inputs, dec_inputs, weights = get_batch()
            _, loss, _ = model.step(session, enc_inputs, dec_inputs, weights, 0, False)
            
            if step % 1000 == 1:
            #if step % 1 == 1:
                print('* step:', step, 'loss:', loss)
                validate_model(text, model, session)
        
        print('*** evaluation! loss:', loss)
        validate_model(text, model, session)


#%time reverse_text(15000)
reverse_text(15000)
#get_ipython().magic(u'time reverse_text(1001)')


#tf.reset_default_graph()
##%time reverse_text(30000)
#get_ipython().magic(u'time reverse_text(30000)')

