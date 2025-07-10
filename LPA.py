import collections
import random
import time 
import networkx as nx
import matplotlib.pyplot as plt

class LPA:
    def __init__(self,G,max_iter=20):
        self._G=G
        self._n=len(G.nodes(False))
        self._max_iter=max_iter

    def can_stop(self):
        for i in range(self._n):
            node=self._G.nodes[i]
            label=node["label"]
            max_label=self.get_max_neighbor_label(i) #周围节点所含有的数量最多的标签
            if label not in max_label:
                return False
            return True
    
    def get_max_neighbor_label(self,node_index):
        m=collections.defaultdict(int) #定义默认值为int（0）的字典
        for neighbor_index in self._G.neighbors(node_index): #对于某个节点的周围节点索引
            neighbor_label=self._G.nodes[neighbor_index]["label"] #找到它的label
            m[neighbor_label]+=1 #更新记录每个标签出现次数的字典
        max_v=max(m.values())
        return [item[0] for item in m.items() if item[1]==max_v] #item[0]是键，item[1]是值，返回值最大的键的列表
    
    def populate_label(self): #更新标签
        visitSequence=random.sample(list(self._G.nodes()),len(self._G.nodes())) #从第一个参数中 无放回地随机抽取第二个参数个元素，返回一个新列表。
        for i in visitSequence:
            node=self._G.nodes[i]
            label=node["label"]
            max_labels=self.get_max_neighbor_label(i)
            if label not in max_labels:
                newlabel=random.choice(max_labels)
                node["label"]=newlabel
    
    def get_communities(self):
        communities=collections.defaultdict(lambda:[]) #定义一个值为空列表的字典
        for node in self._G.nodes(True):
            label=node[1]["label"]
            communities[label].append(node[0]) #社区的键是标签，值是每个标签对应的节点集合
        return communities.values()
    
    def execute(self):
        for i in range(self._n):
            self._G.nodes[i]["label"]=i
        iter_time=0

        while (not self.can_stop() and iter_time < self._max_iter):
            self.populate_label()
            iter_time+=1
        return self.get_communities()
    
 
###可视化###
# 可视化划分结果
def showCommunity(G, partition, pos):
    # 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
    cluster = {}
    labels = {}
    for index, item in enumerate(partition):
        for nodeID in item:
            labels[nodeID] = r'$' + str(nodeID) + '$'  # 设置可视化label
            cluster[nodeID] = index  # 节点分区号

    # 可视化节点
    colors = ['r', 'g', 'b', 'y', 'm']
    shapes = ['v', 'D', 'o', '^', '<']
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=colors[index % len(colors)],
                               node_shape=shapes[index % len(shapes)],
                               node_size=350,
                               alpha=1)

    # 可视化边
    edges = {len(partition): []}
    for link in G.edges():
        # cluster间的link
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            # cluster内的link
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)

    for index, edgelist in enumerate(edges.values()):
        # cluster内
        if index < len(partition):
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color=colors[index % len(colors)])
        else:
            # cluster间
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=3, alpha=0.8, edge_color=colors[index % len(colors)])

    # 可视化label
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.axis('off')
    plt.show()
###

#计算模块度
def cal_Q(partition,G):
    m=len(G.edges(None,False)) # 计算图 G 中所有边的数量（不区分是否有方向）.G.edges()：这是 NetworkX 图对象的方法，用于返回图中的所有边。参数 None：表示不筛选任何节点（即包含所有节点的边）。参数 False：表示不返回边的数据属性（仅返回边的元组，如 (u, v)）。len(...)：统计返回的边列表的长度，即图中边的总数 m。
    a=[]
    e=[]
    for community in partition:
        t=0.0
        for node in community:
            t+=len([x for x in G.neighbors(node) ])
        a.append(t/(2*m))
    for community in partition:
        t=0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if(G.has_edge(community[i],community[j])):
                    t+=1.0
        e.append(t/(2*m))
    q=0.0
    for ei, ai in zip(e,a):
        q+=(ei-ai**2)
    return q

if __name__ == '__main__':
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)
    start_time = time.time()
    algorithm = LPA(G)
    communities = algorithm.execute()
    end_time = time.time()
    for community in communities:
        print(community)

    print(f'模块度{cal_Q(communities, G)}')
    print(f'算法执行时间{end_time - start_time}')
    # 可视化结果
    showCommunity(G, communities, pos)