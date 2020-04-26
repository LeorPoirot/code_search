import numpy as np
from collections import defaultdict
import re
from utils.util import write_pkl, load_pkl


class Dict(object):
    def __init__(self, lddict=None, data=None, lower=True, istree=False, vocab_num=10000):

        self.idxToLabel = {0: '<blank>'}
        self.labelToIdx = {'<blank>': 0}
        self.frequencies = defaultdict(int)
        self.lower = lower
        if lddict:
            self.loadFile(lddict)
        elif istree and data:
            self.build(data, vocab_num, istree)
        elif not istree and data:
            self.build(data, vocab_num)
        else: print('building failed, please set data or lddict!!!')

    def size(self):
        return len(self.idxToLabel)

    def loadFile(self, filename):
        dct = load_pkl(filename)
        self.labelToIdx = dct[0]
        self.idxToLabel = dct[1]

    def writeFile(self, filename):
        dct = list()
        dct.append(self.labelToIdx)
        dct.append(self.idxToLabel)
        write_pkl(dct, filename)

    def lookup(self, key):
        key = key.lower() if self.lower else key
        key = self.filter(key)
        if key not in self.labelToIdx:
            return self.labelToIdx['<blank>']
        else: return self.labelToIdx[key]

    def getLabel(self, idx):
        if idx not in self.idxToLabel:
            return self.labelToIdx[0]
        else:
            return self.labelToIdx[idx]

    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            idx = len(self.idxToLabel)
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx

    def convertToIdx(self, labels, length=-1):
        new_labels = list()
        if length > 0 and length <= len(labels):
            new_labels = labels[:length]
        if length > 0 and length > len(labels):
            new_labels = ['<blank>' for _ in range(length)]
            new_labels[:len(labels)] = labels
        return [self.lookup(i) for i in new_labels]

    def convertToLabels(self, indices):
        return [self.getLabel(idx) for idx in indices]

    def build(self, data, vocab_num, istree=False):
        if not istree:
            for item in data:
                for i in item:
                    i = i.lower() if self.lower else i
                    self.frequencies[i] += 1
        else:
            for item in data:
                self.traverse(item)

        f_pairs = sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)[:vocab_num]
        for pair in f_pairs:
            self.add(pair[0])

    def traverse(self, data):
        name = data['nodeName'].lower() if self.lower else data['nodeName']
        name_filtered = self.filter(name)
        if name_filtered:
            self.frequencies[name_filtered] += 1
        if data['children']:
            self.traverse(data['children'][0])
            self.traverse(data['children'][1])

    def filter(self, node_name):
        flag = re.match(r'([_<>\d\w]+\.)*[_<>\d\w]+$', node_name)
        return node_name.split('.')[-1] if flag else None








