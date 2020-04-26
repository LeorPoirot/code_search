import json
import pickle
import gzip
import jsonlines
import csv
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def un_gz(file_name):
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


def load_json(path):
    with open(path, 'rb') as f:
        jsn = json.load(f)
    return jsn


def load_jsonl(path):
    result = list()
    with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            result.append(item)
    return result


def load_pkl(path):
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.reader(f))

def write_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def init_xavier_linear(opt, linear, init_bias=True, gain=1):
    import torch
    torch.nn.init.xavier_uniform_(linear.weight, gain)
    if init_bias:
        if linear.bias is not None:
            linear.bias.data.normal_(std=opt.init_normal_std)


def load_dataframe(data, column):
    series = data[column].values
    return [json.loads(item.replace('\'', '\"')) for item in series]


def camel_split(name):
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower().replace('_', ' ')

def filter_digit_english(line):
    return re.sub(r'[^\x30-\x39, ^\x41-\x5A,^\x61-\x7A]+', ' ', line)

def get_tokens(line):
    tokens = list()
    for token in word_tokenize(line, 'english'):
        if len(token) > 1:
            if '-' in token:
                ts = token.split('-')
                for t in ts:
                    tokens.append(t)
            else:
                tokens.append(token)
    return tokens

def get_stemmed_words(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = list()
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens