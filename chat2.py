
# coding: utf-8

# In[6]:

import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.danijela import data
import data_utils
import s2s

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/danijela/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
x_len = trainX.shape[-1]
y_len = trainY.shape[-1]
batch_size = 64
x_vocab_size = len(metadata['idx2w'])  
y_vocab_size = x_vocab_size
emb_dim = 364
num_layers = 2

model = s2s.S2S(x_len=x_len,
                y_len=y_len,
                x_vocab_size=x_vocab_size,
                y_vocab_size=y_vocab_size,
                ckpt_path='ckpt/dan/',
                emb_dim=emb_dim,
                num_layers=num_layers)

sess = model.restore_last_session()

def sentence_to_indexes(sen):
    words = sen.split(' ')
    
    indices = []
    
    for word in words:
        if word in metadata['w2idx']:
            indices.append(metadata['w2idx'].get(word))
        else:
            indices.append(metadata['w2idx']['unk'])
    
    indices = indices + [0]*(25 - len(indices))
    
    arr = np.array([indices])
    arr = np.transpose(arr)
    return arr

def ask(str):
    input_ = sentence_to_indexes(str)

    #print(input_.shape)
    #print(input_)
    output = model.predict(sess, input_)
    #print(output)
    #test = model.advance_predict(sess, input_)

    for i in output:
        decoded = data_utils.decode(sequence=i, lookup=metadata['idx2w'], separator=' ').split(' ')
        print('q: [{0}]; a: [{1}]'.format(str, ' '.join(decoded)))
        #print(' '.join(decoded))
        
def ask2(str):
    input_ = sentence_to_indexes(str)

    #print(input_.shape)
    #print(input_)
    output = model.predict(sess, input_)
    #print(output)
    #test = model.advance_predict(sess, input_)

    for i in output:
        decoded = data_utils.decode(sequence=i, lookup=metadata['idx2w'], separator=' ').split(' ')
        print('>>> {}'.format(' '.join(decoded)))


# In[ ]:

text = ''

while True:
    text = input('>>> ')
    if 'close' in text:
        break
    ask2(text)

