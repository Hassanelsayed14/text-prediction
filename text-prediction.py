#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import numpy
#import tensorflow
#print(numpy.__version__)
#print(tensorflow.__version__)

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

texts = ["I love Computational Neuroscience", "I love Cyber Security", "I love artificial intelligence", "I love machine learning"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

max_sequence_length = 4

X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        X.append(seq[:i])   
        y.append(seq[i])   

X = pad_sequences(X, maxlen=max_sequence_length, padding='pre')

y = np.array(y)
y = np.eye(len(tokenizer.word_index) + 1)[y]

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_sequence_length))
model.add(LSTM(50))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

input_text = input("Enter 3 words (space-separated): ")

input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)

predicted_word_idx = model.predict(input_sequence).argmax(axis=-1)
predicted_word = tokenizer.index_word[predicted_word_idx[0]]

print(f"Predicted next word: {predicted_word}")

