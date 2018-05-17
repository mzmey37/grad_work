import re
from collections import Counter
from tensorflow import keras
import shutil
import numpy as np

pad_sequences = keras.preprocessing.sequence.pad_sequences
labels_delimetr = '##'


class Tokenizer:

    def __init__(self):
        self.word2idx = {'padding_token': 0}
        self.word_count = Counter()

    def tokenize(self, text, update_words=False):
        tokens = []
        for w in re.findall('\w+', text):
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

            
def read_batch(data_path, batch_idxes, label2idx):
    path = shutil.os.path.join(data_path, 'len' + str(batch_idxes[0]), *map(str, batch_idxes[1:]))
    sequences, label_idxes = [], []
    with open(path, 'r') as f:
        for line in f:
            try:
                label_set, text = line.strip().split('\t', 1)
                label_set = label_set.split(labels_delimetr)
                label_set = [label2idx[l] for l in label_set if l in label2idx]
                label_idxes.append(label_set)
                sequences.append(list(map(int, text.split())))
            except:
                pass
    result_labels = np.zeros([len(sequences), len(label2idx)], dtype='int')
    for i, label_list in enumerate(label_idxes):
        result_labels[i, label_list] = 1
    return pad_sequences(sequences) if len(sequences) > 0 else np.array([[]]), result_labels


def generator(file_idxes, data_path, label2idx):
    for file_idx in file_idxes:
        yield read_batch(data_path, file_idx, label2idx)