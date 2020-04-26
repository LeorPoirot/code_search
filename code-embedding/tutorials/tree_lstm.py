import dgl
from dgl.data.tree import SST
from dgl.data import SSTBatch
import networkx as nx
import matplotlib.pylab as plt
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgl import unbatch

def plot_tree(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=False, node_size=10,
            node_color=[.5, .5, .5], arrowsize=4)
    plt.show()


class TreeLSTM_Cell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTM_Cell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3*h_size, bias=False)
        self.U_iou = nn.Linear(2*h_size, 3*h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3*h_size))
        self.U_f = nn.Linear(2*h_size, 2*h_size)

    # apply_node -> message_func -> reduce_func
    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # (num_nodes, 2, 256)
        h_mailbox = nodes.mailbox['h']
        h_cat = nodes.mailbox['h'].view(h_mailbox.size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*h_mailbox.size())
        c_ = th.sum(f*nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c_}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.sigmoid(u)
        c = i*u + nodes.data['c']
        h = o*th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = TreeLSTM_Cell(x_size, h_size)

    def forward(self, batch, h, c):
        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # (num_nodes, 256)
        embeds = self.embedding(batch.wordid*batch.mask)
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds))*batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        dgl.prop_nodes_topo(g)

        x = unbatch(g)
        y = x[0].nodes[0].data['h']
        h = self.dropout(g.ndata.pop('h'))

        logits = self.linear(h)
        return logits


def batcher(dev):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        mask=batch_trees.ndata['mask'].to(dev),
                        wordid=batch_trees.ndata['x'].to(dev),
                        label=batch_trees.ndata['y'].to(dev))
    return batcher_dev


if __name__ == '__main__':
    device = th.device('cpu')
    x_size = 256
    h_size = 256
    dropout = 0.5
    lr = 0.05
    weight_decay = 1e-4
    epochs = 10
    # 'x' stands for word in form of int, 'y' stands for the labels,
    trainset = SST(mode='tiny') # 5
    tiny_sst = trainset.trees # 5 dglGraph list
    num_vocabs = trainset.num_vocabs # 19536
    num_classes = trainset.num_classes # 5 classes


    vocab = trainset.vocab # orderedDict([(word: int)]) 19536
    print(len(vocab))
    inv_vocab = {v: k for k, v in vocab.items()}

    a_tree = tiny_sst[0]
    # print(a_tree.ndata['x'])
    # print(a_tree.ndata['x'].tolist())
    res = []
    for token in a_tree.ndata['x'].tolist():
        if token != trainset.PAD_WORD:
            # print(inv_vocab[token], end=' ')
            res.append(inv_vocab[token])
    print(res)
    print(len(res))
    # print(a_tree.edges())
    # print(a_tree.nodes())
    # graph = dgl.batch(tiny_sst)
    # plot_tree(graph.to_networkx())
    model = TreeLSTM(trainset.num_vocabs,
                     x_size,
                     h_size,
                     trainset.num_classes,
                     dropout)
    print(model)
    train_loader = DataLoader(dataset=tiny_sst,
                              batch_size=5,
                              collate_fn=batcher(device),
                              shuffle=False,
                              num_workers=0)

    optimizer = th.optim.Adagrad(model.parameters(),
                                 lr,
                                 weight_decay=weight_decay)
    # training loop
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            g = batch.graph
            n = g.number_of_nodes()
            h = th.zeros((n, h_size))
            c = th.zeros((n, h_size))
            logits = model(batch, h, c)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label, reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = th.argmax(logits, 1)
            acc = float(th.sum(th.eq(batch.label, pred)))/len(batch.label)
            print('Epoch {:05d} | step {:05d} | loss {:.4f} | acc {:.4f}'.format(
                epoch, step, loss.item(), acc
            ))
