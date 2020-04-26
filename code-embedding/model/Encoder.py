import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import Dropout, LSTM, GRU
from utils.util import init_xavier_linear


class BiRNN(nn.Module):
    def __init__(self, opt, dicts):
        super(BiRNN, self).__init__()
        self.dictsz = dicts.size()
        self.opt = opt
        self.nlayers = opt.nlayers
        self.rnn_type = opt.rnn_type
        self.nhid = opt.nhid
        self.outsz = 2*opt.nhid
        self.embsz = opt.embsz
        self.bachsz = opt.batchsz
        self.wemb = nn.Embedding(self.dictsz, self.embsz, padding_idx=0) # esize 300;0
        self.hidden = (torch.autograd.Variable(torch.zeros(2*self.nlayers, self.bachsz, self.nhid)),
                       torch.autograd.Variable(torch.zeros(2*self.nlayers, self.bachsz, self.nhid)))
        if self.opt.init_type in ["xulu"]:
            init_xavier_linear(opt, self.wemb, init_bias=False)

        self.rnn = getattr(nn, opt.rnn_type)(self.embsz,
                                             self.nhid,
                                             self.nlayers,
                                             batch_first=True,
                                             bidirectional=True)

    def forward(self, seq, hidden=None):
        if hidden: self.hidden = hidden
        seq_emb = self.wemb(seq)
        seq_emb = Dropout(0.25)(seq_emb)
        output, self.hidden = self.rnn(seq_emb, self.hidden)
        output = torch.max(output, 1)[0]
        output = torch.tanh(output)

        return output



class MPL(nn.Module):
    def __init__(self, opt, dicts):
        super(MPL, self).__init__()
        self.dictsz = dicts.size()
        self.opt = opt
        self.embsz = opt.embsz
        self.wemb = nn.Embedding(self.dictsz, self.embsz, padding_idx=0)
        if self.opt.init_type in ["xulu"]:
            init_xavier_linear(opt, self.wemb, init_bias=False)

    def forward(self, seq):
        seq_emb = self.wemb(seq)
        seq_emb = Dropout(0.25)(seq_emb)
        output = torch.max(seq_emb, 1)[0]  # for the shape returned contains v and indices
        output = torch.tanh(output)
        return output


class CodeModule(nn.Module):
    def __init__(self, modules, nhid, nout):
        super(CodeModule, self).__init__()
        self.modules = modules
        self.nhid = nhid
        self.nout = nout
        self.dense = nn.Linear(self.nhid, self.nout)

    def forward(self, data_sets):
        output = list()
        for i in range(len(data_sets)):
            output.append(self.modules[i](data_sets[i]))
        output = torch.cat(tuple(output), 1)
        output = self.dense(output)
        return output


class TreeLSTM_Cell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTM_Cell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3*h_size, bias=False)
        self.U_iou = nn.Linear(2*h_size, 3*h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3*h_size))
        self.U_f = nn.Linear(2*h_size, 2*h_size)

    # apply_node -> message_func -> reduce_func
    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # (num_nodes, 2, 256)
        h_mailbox = nodes.mailbox['h']
        h_cat = nodes.mailbox['h'].view(h_mailbox.size(0), -1)
        f = torch.sigmoid(self.U_f(h_cat)).view(*h_mailbox.size())
        c_ = torch.sum(f*nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c_}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(u)
        c = i*u + nodes.data['c']
        h = o*torch.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self, opt, adt_dict, pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.nhid = opt.nhid
        self.embsz = opt.embsz
        self.dictsz = adt_dict.size()
        self.embedding = nn.Embedding(self.dictsz, self.embsz, padding_idx=0)
        if pretrained_emb:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.cell = TreeLSTM_Cell(self.embsz, self.nhid)

    def forward(self, batch):
        g = batch.graph
        n = g.number_of_nodes()
        h = torch.zeros((n, self.nhid))
        c = torch.zeros((n, self.nhid))

        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        dgl.prop_nodes_topo(g)
        un_g = dgl.unbatch(g)
        root_list = list()
        for gh in un_g:
            root_list.append(gh.nodes[0].data['h'])
        output = torch.cat(root_list, 0)
        self.dropout(g.ndata.pop('h'))
        return output