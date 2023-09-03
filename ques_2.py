import random
import numpy as np
import tensorflow as tf
import re
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
def preprocess_text(text):
    text = text.lower()
    text = ' '.join(word_tokenize(text))  
    return text

# Reading of the data from file/corpus
with open("Auguste_Maquet.txt", "r", encoding="utf-8") as f:
    corpus_text = f.read()

# Preprocessing of the corpus (Auguste_Maquet.txt)
corpus_tokens = preprocess_text(corpus_text)
sentences = corpus_tokens.split('.')
random.shuffle(sentences)

train_size = 10000
validation_size = 10000
test_size = 20000

# training data set
train_data = sentences[:train_size]
validation_data = sentences[train_size:train_size+validation_size]
test_data = sentences[train_size+validation_size:train_size+validation_size+test_size]
train_sequences = train_data
validation_sequences = validation_data
test_sequences = test_data

#  5-gram embeddor
seq_length = 5  

# N gram for LSTM
def ngram_seq(sequences, seq_length):
    ngram_seq = []
    for sequence in sequences:
        for i in range(seq_length, len(sequence)):
            ngram = sequence[i - seq_length:i]
            ngram_seq.append(ngram)
    return ngram_seq

train_ngram_seq = ngram_seq(train_sequences, seq_length)
validation_ngram_seq = ngram_seq(validation_sequences, seq_length)
test_ngram_seq = ngram_seq(test_sequences, seq_length)

X_train = np.array([sequence[:-1] for sequence in train_ngram_seq])
y_train = np.array([sequence[-1] for sequence in train_ngram_seq])

X_validation = np.array([sequence[:-1] for sequence in validation_ngram_seq])
y_validation = np.array([sequence[-1] for sequence in validation_ngram_seq])

# Cross-entropy loss
def calculate_perplexity(predictions, targets):
    cross_entropy = -np.log(predictions[np.arange(len(predictions)), targets])        
    perplexity = 2 ** np.mean(cross_entropy)
    return perplexity

#Softmax Embeddings
embedding_dim = 100
hidden_dim = 128
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length - 1),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 64
epochs = 10

model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=batch_size, epochs=epochs)

X_test = np.array([sequence[:-1] for sequence in test_ngram_seq])
y_test = np.array([sequence[-1] for sequence in test_ngram_seq])
predictions = model.predict(X_test)
perp = calculate_perplexity(predictions, y_test)
print("Perplexity is :", perp)