import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl

N = 100  # number of nodes
DAMP = 0.85  # damping factor阻尼因子
K = 10  # number of iterations
g = nx.nx.erdos_renyi_graph(N, 0.1)  # 图随机生成器，生成nx图
g = dgl.DGLGraph(g)  # 转换成DGL图
# nx.draw(g.to_networkx(), node_size=50, node_color=[[.5, .5, .5, ]])  # 使用nx绘制，设置节点大小及灰度值
# plt.show()

g.ndata['pv'] = torch.ones(N) / N  #初始化PageRank值 batch processing
g.ndata['deg'] = g.out_degrees(g.nodes()).float()  #初始化节点特征
print(g.ndata)


#定义message函数，它将每个节点的PageRank值除以其out-degree，并将结果作为消息传递给它的邻居：
def pagerank_message_func(edges):
    pv = edges.src['pv']
    deg = edges.src['deg']
    return {'pv': pv / deg}


#定义reduce函数，它从mailbox中删除并聚合message，并计算其新的PageRank值：
def pagerank_reduce_func(nodes):
    mail_box = nodes.mailbox['pv']
    msgs = torch.sum(mail_box, dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pv': pv}

g.register_message_func(pagerank_message_func)
g.register_reduce_func(pagerank_reduce_func)

def pagerank_naive(g):
    # Phase #1: 沿所有边缘发送消息。
    print(g.edges())
    for u, v in zip(*g.edges()):
        g.send((u, v))
    # Phase #2: 接收消息以计算新的PageRank值。
    for v in g.nodes():
        g.recv(v)


pagerank_naive(g)