import nltk
from nltk.tokenize import word_tokenize
from nltk.lm import NgramCounter
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import Laplace
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
nltk.download('punkt')

# Cleaning of data,tokenising into words
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
num_samples = len(sentences)
num_val = 10000  
num_test = 20000
validation_set = sentences[:num_val]
test_set = sentences[num_val:num_val + num_test]
training_set = sentences[num_val + num_test:]

def tokenize_sentence(sentence):
    return word_tokenize(sentence)

validation_set = [tokenize_sentence(sentence) for sentence in validation_set]
test_set = [tokenize_sentence(sentence) for sentence in test_set]
training_set = [tokenize_sentence(sentence) for sentence in training_set]

n = 5  # N gram order 
train_data, padded_sents = padded_everygram_pipeline(n, training_set)
model = Laplace(n)
model.fit(train_data, padded_sents)
val_data, val_padded_sents = padded_everygram_pipeline(n, validation_set)
perp = model.perplexity(val_padded_sents)
print(f"Perplexity is: {perp:.6f}")

dropout_rates = [0.2, 0.4, 0.6]
layer_dimensions = [(128, 128), (256, 128), (128, 256)]
optimizers = ['adam', 'rmsprop', 'sgd']
ave_tra_perp = {}
ave_test_perp = {}
# Iteration of the 
for dropout_rate in dropout_rates:
    for dim1, dim2 in layer_dimensions:
        for optimizer in optimizers:
            avg_train_perplexity = perplexity
            avg_test_perplexity = perplexity
            
            hyp_tuple = (dropout_rate, (dim1, dim2), optimizer)
            ave_tra_perp[hyp_tuple] = avg_train_perplexity
            ave_test_perp[hyp_tuple] = avg_test_perplexity
            
            
optimal_hyperparameters = min(ave_test_perp, key=ave_test_perp.get)
print("Most optimal hyperparameters:", optimal_hyperparameters)

#plotting of various graphs showing the perplexities
plt.plot(dropout_rates)
plt.show()
plt.plot(layer_dimensions)
plt.show()
plt.plot(optimizers)
plt.show() 