
# coding: utf-8

# In[4]:

import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.ana import data
import data_utils

path_x = 'datasets/ana'

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/ana/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
x_len = trainX.shape[-1]
y_len = trainY.shape[-1]
batch_size = 64
x_vocab_size = len(metadata['idx2w'])  
y_vocab_size = x_vocab_size
emb_dim = 364
num_layers = 2

import s2s

# In[7]:

model = s2s.S2S(x_len=x_len,
                y_len=y_len,
                x_vocab_size=x_vocab_size,
                y_vocab_size=y_vocab_size,
                ckpt_path='ckpt/ana/',
                emb_dim=emb_dim,
                num_layers=num_layers)


# In[ ]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)