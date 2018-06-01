import re
from collections import Counter
import shutil
from hashlib import sha512
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm_notebook


LABELS_DELIMETR = '##'
FIELDS_DELIMETR = '\t\t'

# Merges tokenizer with words from pretrained set and returns vectors for them

def merge_with_pretrained(tokenizer, embed_path, embedding_dim):
    word2vec = {}
    with open(embed_path, 'r') as fwv:
        for line in tqdm_notebook(fwv, desc='read vectors'):
            word, vec = line.strip().split('\t')
            vec = np.array(vec.split(), 'float32')
            word2vec[word] = vec

    words2remove = set()
    for word in tqdm_notebook(tokenizer.word_count, desc='looking for words to remove'):
        if word not in word2vec:
            words2remove.add(word)

    for word in tqdm_notebook(words2remove, desc='removing words'):
        tokenizer.word2idx.pop(word)
        tokenizer.word_count.pop(word)
    tokenizer.shrink(len(tokenizer.word_count))

    word_vectors = np.zeros([len(tokenizer.word2idx), embedding_dim], dtype='float32')
    for word, idx in tokenizer.word2idx.items():
        if word in word2vec:
            word_vectors[idx] = word2vec[word]
    return word_vectors


# Reading batches
        
def read_batch(path):
    labels, texts = [], []
    for line in open(path, 'r'):
        line_labels, text = line.strip().split(FIELDS_DELIMETR)
        labels.append(line_labels.split(LABELS_DELIMETR))
        texts.append(text)
    return texts, labels


def iterate_raw_batches(file_idxes, p_data):
    for pad_idx, dir_idx, batch_idx in file_idxes:
        yield read_batch(shutil.os.path.join(
            p_data, 'len' + str(pad_idx), str(dir_idx), str(batch_idx)))


def map_labels(label_list, label2idx):
    return [label2idx[l] for l in label_list if l in label2idx]


def iterate_batches(file_idxes, tokenizer, label2idx, p_data):
    for batch_x, batch_y in iterate_raw_batches(file_idxes, p_data):
        batch_y = [map_labels(label_list, label2idx) for label_list in batch_y]
        batch_x = [tokenizer.tokenize(t) for t in batch_x]
        yield batch_x, batch_y


# Writing batches

def write_batch(batch, path, filtered_labels, filtered_texts):
    """Writes batch of idxes into file path"""
    with open(path, 'w') as f:
        for idx in batch:
            f.write(FIELDS_DELIMETR.join([filtered_labels[idx], filtered_texts[idx]]) + '\n')


def get_dir_path(path, *idxes):
    return shutil.os.path.join(path, *map(str, idxes))


def write_batches(idxes, path, batch_size, filtered_labels, filtered_texts, files_in_folder):
    """Functions to write batches in format:
    - data_path
        - len{pad_len}
            - {number_of_dir_inside_batches_with_such_len}
                - {idx_of_batch}
    Inside file:
    {label1}##{label_2}##...##{label_n}\t\t{sentence}"""

    dir_idx = 0
    shutil.os.mkdir(get_dir_path(path, dir_idx))
    for batch_idx, start_idx in enumerate(range(0, len(idxes), batch_size)):
        end_idx = min(start_idx + batch_size, len(idxes))
        if len(shutil.os.listdir(get_dir_path(path, dir_idx))) == files_in_folder:
            dir_idx += 1
            shutil.os.mkdir(get_dir_path(path, dir_idx))
        write_batch(idxes[start_idx : end_idx], get_dir_path(path, dir_idx, batch_idx),
                    filtered_labels, filtered_texts)

        
# Tokenizer with collecting dictionary
        
class Tokenizer:
    
    def __init__(self):
        self.word2idx = {'padding_token': 0}
        self.word_count = Counter()
        
    def _get_words(self, text):
        return re.sub(
            '[.,?;*!%^&_+():-\[\]{}]', '',
            text.replace('"', '').\
            replace('/', '').\
            replace('\\', '').\
            replace("'",'').\
            strip().\
            lower()).split()

    def tokenize(self, text, update_words=False):
        tokens = []
        for w in self._get_words(text):
            if update_words:
                if w not in self.word_count:
                    self.word2idx[w] = len(self.word2idx)
                self.word_count.update([w])
                tokens.append(self.word2idx[w])
            else:
                if w in self.word_count:
                    tokens.append(self.word2idx[w])
        return tokens
    
    def shrink(self, n):
        self.word2idx = {'padding_token': 0}
        best_words = self.word_count.most_common(n)
        self.word_count = Counter()
        for w, c in best_words:
            self.word2idx[w] = len(self.word2idx)
            self.word_count[w] = c

            
# Small support


def get_pad_lens(lens, n_splits=40):
    percentiles = np.percentile(lens, np.linspace(0, 100, n_splits + 1)[1:]).astype('int')
    sorted_pad_lens = np.sort(percentiles)
    percentiles = np.repeat([percentiles], len(lens), axis=0)
    return sorted_pad_lens[np.argmax(np.greater_equal(percentiles, lens.reshape(-1, 1)), axis=1)]

def unpickle_obj(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)


def pickle_obj(obj, path):
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)
        
def get_hash(text):
    return sha512(text.encode()).hexdigest()

