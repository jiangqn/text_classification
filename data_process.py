#encoding=utf-8
import numpy as np
import random
from nltk.tokenize import WordPunctTokenizer
import pickle

def word_tokenizer(sentence):
    return WordPunctTokenizer().tokenize(sentence)

file = open('./dataset/corpus.txt', 'r', encoding=u'utf-8', errors='ignore')
data = file.read().split('\n')
random.shuffle(data)

texts_buf = []
labels_buf = []
max_len = 0
for example in data:
    content = word_tokenizer(example.lower())
    labels_buf.append(content[0])
    texts_buf.append(content[1:])

vocab = dict()
for text in texts_buf:
    max_len = max(max_len, len(text))
    for word in text:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

vocab_list = list(vocab.items())
vocab_list.sort(key=lambda x: x[1], reverse=True)
vocab = dict()
vocab['UNK'] = 0

for i, item in enumerate(vocab_list):
    vocab[item[0]] = i + 1
vocab_size = len(vocab)
data_size = len(texts_buf)

texts = np.zeros((data_size, max_len), dtype=np.int32)
labels = np.zeros(data_size, dtype=np.int32)

for i in range(data_size):
    for j, word in enumerate(texts_buf[i]):
        texts[i][j] = vocab[texts_buf[i][j]]
    labels[i] = int(labels_buf[i][9]) - 1

texts = np.array(texts)
labels = np.array(labels)

params = {
    'vocab_size': vocab_size,
    'class_num': 2,
    'l2_reg_lambda': 1e-3,
    'train_texts': texts[0: 7000],
    'train_labels': labels[0: 7000],
    'val_texts': texts[7000: 8500],
    'val_labels': labels[7000: 8500],
    'test_texts': texts[8500: 10000],
    'test_labels': labels[8500: 10000]
}

with open('./dataset/params.pickle', 'wb') as handle:
    pickle.dump(params, handle)
