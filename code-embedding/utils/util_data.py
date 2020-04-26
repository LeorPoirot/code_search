from torch.utils.data import Dataset, DataLoader
from data.Dict import Dict
from collections import namedtuple
from utils.util import load_pkl, load_dataframe
import pandas as pd
import os
from torch import tensor
from model.Encoder import MPL, BiRNN, CodeModule, TreeLSTM
from model.CodeRetrievalModel import CodeRetrievalModel
import json
import copy
import dgl
import networkx as nx
from dgl.data import SSTBatch
import torch


class WrapDataLoader():
    def __init__(self, deepcs_loader, tree_loader):
        self.deepcs_loader = deepcs_loader
        self.tree_loader = tree_loader

    def __iter__(self):
        self.deepcs_loader = iter(self.deepcs_loader)
        self.tree_loader = iter(self.tree_loader)
        return self

    def __next__(self):
        return next(self.deepcs_loader), next(self.tree_loader)

    def __len__(self):
        return len(self.deepcs_loader)


class DeepCSDateSet(Dataset):
    def __init__(self, *data_set):
        super(DeepCSDateSet, self).__init__()
        self.data_set = data_set
        self.length = 0
        len_list = [len(data) for data in data_set]
        len_set = set(len_list)
        assert len(len_set) == 1
        self.length = len_list[0]

    def __getitem__(self, index):
        return [data[index] for data in self.data_set]

    def __len__(self):
        return self.length


def my_collate(batch):
    num = len(batch[0])
    result = list()
    for n in range(num):
        data = tensor([item[n] for item in batch])
        if torch.cuda.is_available():
            data = data.cuda()
        result.append(data)
    return result


def build_dict(data):

    tree_json = parse_ast(data)
    dict_tree = Dict(data=tree_json, istree=True)
    dict_tree.writeFile('./data/java/deep_with_ast/ast_dict.pkl')
    print('ast_dict was builded!!!')
    meth_name = load_dataframe(data, 'method_name')
    dict_meth = Dict(data=meth_name)
    dict_meth.writeFile('./data/java/deep_with_ast/meth_dict.pkl')
    print('meth_dict was builded!!!')
    tokens = load_dataframe(data, 'tokens')
    dict_token = Dict(data=tokens)
    dict_token.writeFile('./data/java/deep_with_ast/token_dict.pkl')
    print('token_dict was builded!!!')
    apis = load_dataframe(data, 'api_seq')
    dict_api = Dict(data=apis)
    dict_api.writeFile('./data/java/deep_with_ast/api_dict.pkl')
    print('api_dict was builded!!!')
    comments = load_dataframe(data, 'desc')
    dict_comment = Dict(data=comments)
    dict_comment.writeFile('./data/java/deep_with_ast/comment_dict.pkl')
    print('comment_dict was builded!!!')


def parse_ast(data):
    tree = data['ast'].tolist()
    tree_json = list()
    for item in tree:
        if item == '{"nodeName":"BlockStmt", mask:"1", children:[]}' or len(item) > 1000:
            tree_json.append(json.loads('{"nodeName":"BlockStmt", "mask":"1", "children":[]}'))
        else:
            tree_json.append(json.loads(item))
    return tree_json


def load_dict():
    dict_dir = './data/java/deep_with_ast/dict'
    if not os.listdir(dict_dir):
        data = pd.read_csv('./data/java/java_with_ast_0.csv')
        build_dict(data)

    dict_meth = load_pkl('./data/java/deep_with_ast/dict/meth_dict.pkl')
    dict_token = load_pkl('./data/java/deep_with_ast/dict/token_dict.pkl')
    dict_api = load_pkl('./data/java/deep_with_ast/dict/api_dict.pkl')
    dict_comment = load_pkl('./data/java/deep_with_ast/dict/comment_dict.pkl')
    return dict_meth, dict_token, dict_api, dict_comment


def load_data(opt, model='train'):
    data = pd.read_csv('./data/java/deep_with_ast/java_with_ast_0.csv')
    # build_dict(data)
    meths = load_dataframe(data, 'method_name')
    dict_meth = Dict(lddict='./data/java/deep_with_ast/dict/meth_dict.pkl')
    tokens = load_dataframe(data, 'tokens')
    dict_token = Dict(lddict='./data/java/deep_with_ast/dict/token_dict.pkl')
    apis = load_dataframe(data, 'api_seq')
    dict_api = Dict(lddict='./data/java/deep_with_ast/dict/api_dict.pkl')
    comments = load_dataframe(data, 'desc')
    dict_comment = Dict(lddict='./data/java/deep_with_ast/dict/comment_dict.pkl')

    dict_ast = Dict(lddict='./data/java/deep_with_ast/dict/ast_dict.pkl')

    meth_data = [dict_meth.convertToIdx(meth, length=6) for meth in meths]
    token_data = [dict_token.convertToIdx(token, length=50) for token in tokens]
    api_data = [dict_api.convertToIdx(api, length=30) for api in apis]
    comment_data = [dict_comment.convertToIdx(comment, length=30) for comment in comments]
    deepcs_dataset = DeepCSDateSet(meth_data, token_data, api_data, comment_data)

    tree_json = parse_ast(data)
    ast_data = get_tree_dataset(tree_json, dict_ast)

    if torch.cuda.is_available():
        device = torch.device('gpu')
    else: device = torch.device('cpu')

    tree_loader = DataLoader(dataset=ast_data,
                              batch_size=opt.batchsz,
                              collate_fn=batcher(device),
                              shuffle=False,
                              num_workers=2)

    loader = DataLoader(
        dataset=deepcs_dataset,
        batch_size=opt.batchsz,
        shuffle=False,
        num_workers=2,
        collate_fn=my_collate)

    wrapDataLoader = WrapDataLoader(loader, tree_loader)

    NamedDict = namedtuple('NamedDict', ['meth_name', 'tokens', 'api_seq', 'description', 'ast'])
    all_dict = NamedDict(dict_meth, dict_token, dict_api, dict_comment, dict_ast)

    if model == 'train':
        return wrapDataLoader, all_dict

    if model == 'n_query':
        code_body = data['original_string'].tolist()
        func_name = data['func_name'].tolist()
        return wrapDataLoader, all_dict, code_body, func_name

def create_code_retrieval_model(opt, all_dict):
    meth_module = BiRNN(opt, all_dict.meth_name)
    api_module = BiRNN(opt, all_dict.api_seq)
    token_module = MPL(opt, all_dict.tokens)
    ast_module = TreeLSTM(opt, all_dict.ast)
    merge_size = meth_module.outsz + api_module.outsz + token_module.embsz + ast_module.nhid
    code_encoder = CodeModule((meth_module, api_module, token_module, ast_module), merge_size, opt.nout)
    comment_encoder = BiRNN(opt, all_dict.description)
    model = CodeRetrievalModel(code_encoder, comment_encoder, opt)

    if torch.cuda.is_available():
        meth_module = meth_module.cuda()
        api_module = api_module.cuda()
        token_module = token_module.cuda()
        ast_module = ast_module.cuda()
        comment_encoder = comment_encoder.cuda()
        code_encoder = CodeModule((meth_module, api_module, token_module, ast_module), merge_size, opt.nout).cuda()
        model = CodeRetrievalModel(code_encoder, comment_encoder, opt).cuda()
    return model


def get_tree_dataset(trees, dict_tree):
    data_tree_dgl_graphs = list()
    for t_json in trees:
        tree_dgl_graph = build_tree(t_json, dict_tree)
        data_tree_dgl_graphs.append(tree_dgl_graph)
    return data_tree_dgl_graphs


def build_tree(tree_json, dict_ast):
    g = nx.DiGraph()
    def _rec_build(nid, t_json):

        children = t_json['children']
        node_name = tree_json['nodeName']
        node_idx = dict_ast.lookup(node_name)

        if len(children) == 2:
            cid = g.number_of_nodes()
            g.add_node(cid, x=node_idx, y=node_idx, mask=0)

            for c in children:
                cid = g.number_of_nodes()
                _rec_build(cid, c)
                g.add_edge(cid, nid)
        else:
            assert len(children) == 0
            cid = g.number_of_nodes()
            g.add_node(cid, x=node_idx, y=node_idx, mask=1)

    _rec_build(0, tree_json)
    ret = dgl.DGLGraph()

    ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
    return ret


def batcher(dev):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        mask=batch_trees.ndata['mask'].to(dev),
                        wordid=batch_trees.ndata['x'].to(dev),
                        label=batch_trees.ndata['y'].to(dev))
    return batcher_dev



if __name__ == '__main__':
    data = pd.read_csv('../data/java/temp/mini_train.csv')
    tree = data['ast'].tolist()
    tree_json = [json.loads(item) for item in tree]
    dict_tree = Dict(data=tree_json, istree=True)
    ret = get_tree_dataset(tree_json, dict_tree)
    device = torch.device('cpu')
    train_loader = DataLoader(dataset=ret,
                              batch_size=5,
                              collate_fn=batcher(device),
                              shuffle=False,
                              num_workers=0)
    for i in train_loader:
        print()






