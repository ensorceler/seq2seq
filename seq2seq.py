#!/usr/bin/env python
# coding: utf-8

# In[1]:

#seq2seq model creation
#@author :ensorceler /mel0n 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


data=pd.read_csv("ben.txt",delimiter='\t')
#ata=data.drop('Unnamed: 0',axis=1)
data=data.iloc[:,:2]
data.columns=['Q1','A1']
data=data[:5000]
input_text=list(data['Q1'])
target_text=list(data['A1'].apply(lambda x:x+' <eos>'))
target_text_input=list(data['A1'].apply(lambda x:'<sos> '+x))
data


# In[3]:


MAX_NUM_WORDS=20000
EPOCHS=40
latent_dim=256
embedding_dim=100
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer,one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer_inputs=Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_text)
input_sequences=tokenizer_inputs.texts_to_sequences(input_text)

tokenizer_outputs=Tokenizer(num_words=MAX_NUM_WORDS,filters='')
tokenizer_outputs.fit_on_texts(target_text+target_text_input)
target_sequences=tokenizer_outputs.texts_to_sequences(target_text)
target_sequences_inputs=tokenizer_outputs.texts_to_sequences(target_text_input)


word2idx_inputs=tokenizer_inputs.word_index
word2idx_outputs=tokenizer_outputs.word_index

max_len_input=min(100,max(len(s) for s in input_sequences))
max_len_target=min(100,max(len(s) for s in target_sequences))

print("tokenizing is done and word index found")


# In[4]:


encoder_input_data=pad_sequences(input_sequences,maxlen=max_len_input)
decoder_input_data=pad_sequences(target_sequences_inputs,padding='post',maxlen=max_len_target)

decoder_targets=pad_sequences(target_sequences,padding='post',maxlen=max_len_target)

num_words_input=len(word2idx_inputs)+1
num_words_output=len(word2idx_outputs)+1

print("padding ia done" )

print(encoder_input_data[0])
print(decoder_input_data[0])
print(decoder_targets[0])


# In[5]:


one_hot_targets=np.zeros(
(
len(input_text),
max_len_target,
 num_words_output   
),
dtype='float32'
)

for i,d in enumerate(decoder_targets):
    for t,word in enumerate(d):
        if word!=0:
            one_hot_targets[i,t,word]=1
one_hot_targets.shape


# In[6]:


import os
from tensorflow.keras.layers import Embedding
EMBEDDING_DIM=100

embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  input_length=max_len_input,
  # trainable=True
)


# In[7]:


from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Input,Conv2D,LSTM,GRU,Dense,TimeDistributed,Embedding
from tensorflow.keras.models import Model
encoder_inputs=Input(shape=(max_len_input,))
x=embedding_layer(encoder_inputs)
lstm=CuDNNLSTM(latent_dim,return_state=True)
encoder_outputs,h,c=lstm(x)

encoder_states=[h,c]

decoder_inputs=Input(shape=(max_len_target,))
decoder_embedding=Embedding(num_words_output,EMBEDDING_DIM)
y=decoder_embedding(decoder_inputs)
decoder_lstm=CuDNNLSTM(latent_dim,return_state=True,return_sequences=True)

decoder_outputs,_,_=decoder_lstm(y,initial_state=encoder_states)
decoder_dense=Dense(num_words_output,activation="softmax")

decoder_outputs=decoder_dense(decoder_outputs)

model=Model([encoder_inputs,decoder_inputs],decoder_outputs)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
model.fit([encoder_input_data,decoder_input_data],one_hot_targets,epochs=40,validation_split=0.2,batch_size=32)


# In[9]:


encoder_model=Model(encoder_inputs,encoder_states)
decoder_state_h=Input(shape=(latent_dim,))
decoder_state_c=Input(shape=(latent_dim,))
decoder_state_inputs=[decoder_state_h,decoder_state_c]

decoder_single_input=Input(shape=(1,))
z=decoder_embedding(decoder_single_input)

decoder_outputs,h,c=decoder_lstm(z,initial_state=decoder_state_inputs)

decoder_states=[h,c]
decoder_outputs=decoder_dense(decoder_outputs)

decoder_model=Model([decoder_single_input]+decoder_state_inputs,
                   [decoder_outputs]+decoder_states)


# In[12]:



def seq(input_seq):
    state_val=encoder_model.predict(input_seq)
    target_seq=np.zeros((1,1))
    target_seq[0,0]=word2idx_outputs['<sos>']
    ans=[]
    eos=word2idx_outputs['<eos>']
    for _ in range(max_len_target):
        output_tokens,h,c=decoder_model.predict([target_seq]+state_val)
        
        idx=np.argmax(output_tokens[0,0,:])
        if idx==eos:
            break
        if idx>0:
            word=tokenizer_outputs.index_word[idx]
            ans.append(word)
        target_seq[0,0]=idx
        state_val=[h,c]
    return ans

for i in range(0,100):
    print(input_text[i])
    print("translation-->")
    res=seq(encoder_input_data[i:i+1])
    print(res)

