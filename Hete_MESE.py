import numpy as np
import networkx as nx
from networkx.algorithms import community
from cdlib import algorithms, evaluation
from collections import defaultdict

TYPE_MAPPING = {
    'author': 'A',
    'paper': 'P',
    'venue': 'V',
    'writes': 'P',    # 作者→论文的边类型映射为P
    'be_writen':'A',
    'cited_in':'→P',
    'cites' : 'P←',
    'published_in': 'V',  # 论文→会议的边类型映射为V
    'publish' : 'P',
    'has_keyword' : 'K',
    'keyword_in' : 'P'
}

import networkx as nx

def convert_hetero_graph_to_networkx(hetero_dict):
    """将字典格式的异构图转换为 networkx.MultiDiGraph"""
    G = nx.MultiDiGraph()
    
    # 添加节点并标注类型
    for node_type, nodes in hetero_dict.items():
        if node_type != 'edges':  # 排除edges键
            for node in nodes:
                G.add_node(node, type=node_type)
    
    # 添加边并标注关系类型
    for edge in hetero_dict['edges']:
        u, v, relation_type = edge
        G.add_edge(u, v, relation=relation_type)
    
    return G

def process_nested_community(community_list):
    """
    处理特殊嵌套结构的社区数据
    :param community_list: [{'author': {...}, 'paper': {...}, ...}]
    :return: 节点集合列表
    """
    combine_list=[]
    for community_dict in community_list:
    # if len(community_list) != 1 or not isinstance(community_list[0], dict):
    #     raise ValueError("输入必须为包含单个字典的列表")
    
    # community_dict = community_list[0]
        processed = []
        
        # 方案1：合并所有类型节点到一个社区
        combined = set()
        # 方案2：按类型分开为多个社区
        separated = []
        
        for node_type, nodes in community_dict.items():
            if node_type != 'edges' and nodes:  # 过滤掉edges和空集合
                combined.update(nodes)
                separated.append(nodes)
        combine_list.append(combined)
        
    return combine_list

def build_multiplex_network(hetero_graph, central_node_type, meta_paths):
    """
    构建多路复用网络（每个元路径对应一个网络层）
    :param hetero_graph: 异构网络（需预先构建，格式如 {node_type: [node_list], edges: [(node1, node2, type)]})
    :param central_node_type: 中心节点类型（如 'author'）
    :param meta_paths: 元路径列表（如 ['A-P-A', 'A-P->P-A']）
    :return: 多路复用网络 {layer_name: networkx.Graph}
    """
    multiplex_net = {}
    for path in meta_paths:
        layer_name = f"layer_{path}"#以元路径的名字命名
        G = nx.Graph()#创造一个空的图对象

        # 以下在模拟元路径实例（实际需根据真实数据生成）
        for u in hetero_graph[central_node_type]:
            for v in hetero_graph[central_node_type]:
                if u != v:
                    instances = find_path_instances(u, v, path, hetero_graph)
                    if instances:
                        weight = sum(1/len(path) for path in instances) #权重是路径条数
                        G.add_edge(u, v, weight=weight)
                        multiplex_net[layer_name] = G
                        print(G)
    return multiplex_net

def find_path_instances(u, v, meta_path, graph):
    """严格实现元路径实例查找"""
    path_types = meta_path.replace('←', '-←').replace('→', '-→').split('-')  #处理含有箭头的元路径
    instances = []
    
    def dfs(current, path, remaining_types): #DFS深度优先搜索算法查找实例
            if len(remaining_types) == 0:
                if current == v:
                    instances.append(path)
                return
            rel_type= remaining_types[0]

            for edge in graph['edges']:
                if rel_type == '→P' and edge[0] == current and TYPE_MAPPING[edge[2]] == rel_type:
                    dfs(edge[1], path + [edge], remaining_types[1:])
                elif rel_type == 'P←' and edge[1] == current and TYPE_MAPPING[edge[2]] == rel_type:
                    dfs(edge[0], path + [edge], remaining_types[1:])
                elif edge[0] == current and TYPE_MAPPING[edge[2]] == rel_type:  #映射边的类型和元路径节点
                    dfs(edge[1], path + [edge], remaining_types[1:]) 
    
    dfs(u, [], path_types[1:])  # 第一个类型已由u确定
    return instances

def detect_seed_communities(multiplex_net):
    """
    通过共识图检测种子社区
    :param multiplex_net: 多路复用网络
    :return: 种子社区列表 [set(node1, node2, ...), ...]
    """
    # Step 1: 对每层运行标签传播算法（LPANNI的简化版）
    layer_communities = {}
    for layer_name, G in multiplex_net.items():
        communities = algorithms.label_propagation(G).communities
        layer_communities[layer_name] = communities
    
    # Step 2: 构建共识图（公式3）
    consensus_graph = nx.Graph()
    nodes = list(multiplex_net.values())[0].nodes()  # 所有层节点相同，返回的是第一个图的所有节点
    consensus_graph.add_nodes_from(nodes)
    
    for u in nodes:
        for v in nodes:
            if u != v:
                total_weight = 0
                for layer_name, communities in layer_communities.items():
                    # 检查u和v是否在同一社区
                    same_community = any((u in c) and (v in c) for c in communities)#对每个社区 c，检查 u 和 v 是否同时属于 c，返回一个布尔值序列（True/False）。any()函数如果生成器表达式中至少有一个 True，则返回 True，否则返回 False。
                    if same_community:
                        weight = multiplex_net[layer_name].get_edge_data(u, v, {}).get('weight', 0)#fet_edge_data()可以返回边的数据字典，get("weight",0)提取weight的值，如果没有则返回0
                        total_weight += weight
                if total_weight > 0:
                    consensus_graph.add_edge(u, v, weight=total_weight)
    print(f'共识图：{consensus_graph}')
    
    # Step 3: 在共识图上再次检测社区
    seed_communities = algorithms.louvain(consensus_graph).communities
    print(f'种子社区：{[set(c) for c in seed_communities]}')
    """
    为什么转换为 set？
    高效成员检查：集合（set）的成员检查（x in set）是 O(1) 时间复杂度，而列表（list）是 O(n)。在社区检测中，频繁需要检查节点是否属于某个社区（如 if node in community），用集合更高效。
    去重：如果原始社区中有重复节点（如 ["A", "B", "A"]），转换为集合会自动去重（变为 {"A", "B"}）。
    集合操作：后续可能需要进行集合的交并差操作（如求社区重叠部分），集合类型更方便（如 community1 & community2）。
    """
    return [set(c) for c in seed_communities] #将社区列表中的每个社区转换为set集合形式并以列表形式返回。

def get_neighbors(node, adj):
    """通过邻接表快速查询邻居"""
    return list(adj.get(node, set()))

def seed_expansion(hetero_graph, seed_communities, central_node_type,adjacency_list):
    """
    将种子社区扩展到其他节点类型
    :param hetero_graph: 异构网络
    :param seed_communities: 种子社区列表
    :param central_node_type: 中心节点类型
    :return: 异构社区列表 [{'authors': [], 'papers': [], ...}, ...]
    """
    final_communities = []
    for seed in seed_communities:
        community = {central_node_type: set(seed)}#为当前正在扩展的种子社区创建一个空容器，用于存储该社区内所有类型的节点
        # 初始化其他节点类型
        for node_type in hetero_graph.keys():
            if node_type != central_node_type:#跳过中心节点
                community[node_type] = set()#初始化其他类型节点的社区为空社区
        
        # 扩展非中心节点（公式4）
        for node_type, nodes in hetero_graph.items():#nodes是某个类型对应的所有节点集合
            if node_type == central_node_type:#跳过中心节点
                continue
            for node in nodes:#非中心节点的某种类型节点的每一个节点
                max_sim = -1
                best_seeds = []
                for seed_node in seed:#某个种子社区的每个节点
                    # 计算相似度（简化版：直接统计共同邻居）
                    neighbors = set(get_neighbors(seed_node,adjacency_list))  # 需实现get_neighbors
                    sim = len(neighbors & set(get_neighbors(node,adjacency_list)))
                    if sim > max_sim:
                        max_sim = sim
                        best_seeds = [seed_node]
                    elif sim == max_sim:
                        best_seeds.append(seed_node) #找到和该节点相似值最大的种子节点并加入到best_seeds的队列中
                if max_sim > 0.8:
                    community[node_type].add(node)#如果最大相似度大于0了，就将该节点归入到该类型节点的社区中
        
        final_communities.append(community)
    return final_communities

def HETE_MESE(hetero_graph, central_node_type, meta_paths,adjacency_list):
    # Step 1: 构建多路复用网络
    multiplex_net = build_multiplex_network(hetero_graph, central_node_type, meta_paths)
    
    # Step 2: 检测种子社区
    seed_communities = detect_seed_communities(multiplex_net)
    
    # Step 3: 种子扩展
    final_communities = seed_expansion(hetero_graph, seed_communities, central_node_type,adjacency_list)
    
    return final_communities

def build_adjacency_list(hetero_graph):
    """预处理：构建邻接表 {node: [neighbor1, neighbor2, ...]}"""
    adj = defaultdict(set)
    for (u, v, _) in hetero_graph['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def overlapping_modularity(G, communities):
    """适用于有向多关系图的简化模块度计算（忽略方向性和关系类型）"""
    A = nx.to_numpy_array(G)  # 自动忽略边属性
    m = G.number_of_edges()
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    Q = 0.0

    # 计算O_v（节点所属社区数）
    O = {n: 0 for n in nodes}
    for comm in communities:
        for n in comm:
            O[n] += 1

    # 遍历社区和节点对
    for comm in communities:
        for v in comm:
            for w in comm:
                if v == w:
                    continue
                i, j = node_index[v], node_index[w]
                kv = G.out_degree(v)  # 有向图的出度
                kw = G.in_degree(w)   # 有向图的入度
                Q += (A[i][j] - (kv * kw) / (2 * m)) / (O[v] * O[w])

    return Q / (2 * m)

def evaluate_communities(hetero_graph, communities):
    results={}
    G = convert_hetero_graph_to_networkx(hetero_graph)
    C=process_nested_community(communities)
    print("图中所有节点:", G.nodes())
    M=overlapping_modularity(G, C)
    results['modularity']=M
    
    # 3. 计算关键词相关性
    keyword_coherence = []
    for com in communities:
        papers = com.get('paper', [])
        keywords = defaultdict(int)
        for (u, v, edge_type) in hetero_graph['edges']:
            if edge_type == 'has_keyword' and u in papers:
                keywords[v] += 1
        if keywords:
            avg_coherence = sum(keywords.values()) / len(keywords)
            keyword_coherence.append(avg_coherence)
    results['keyword_coherence'] = np.mean(keyword_coherence) if keyword_coherence else 0
    
    return results

# def evaluate_communities(hetero_graph, communities):
#     """实现论文中的评估指标"""
#     results = {}
    
#     # 1. 重叠模块度Q
#     # try:
#     #     q = evaluation.newman_girvan_modularity(
#     #         nx.Graph([(u,v) for u,v,_ in hetero_graph['edges']]),
#     #         [list(c['author']) for c in communities]
#     #     ).score
#     #     results['modularity'] = q
#     # except:
#     #     results['modularity'] = 0
#     #     print('Something is wrong when calculating the modularity!')

#     modularity = community.modularity(hetero_graph, communities)
#     print(f"Modularity: {modularity}")
    
#     # 2. 社区内关键词相关性 (简化实现)
#     keyword_coherence = []
#     for com in communities:
#         papers = com['paper']
#         keywords = defaultdict(int)
#         for p in papers:
#             for u, v, t in hetero_graph['edges']:
#                 if u == p and t == 'has_keyword':
#                     keywords[v] += 1
#         if len(keywords) > 0:
#             avg_coherence = sum(keywords.values()) / len(keywords)
#             keyword_coherence.append(avg_coherence)
#     results['keyword_coherence'] = np.mean(keyword_coherence) if keyword_coherence else 0
    
#     return results

def run_experiments():
    """运行完整实验"""
    # 构建网络
    hetero_graph = {
        'author': ['a1', 'a2', 'a3', 'a4'],
        'paper': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'venue': ['v1', 'v2'],
        'keyword': ['k1', 'k2', 'k3'],
        'edges': [
            # 作者-论文关系 (写作)
            ('a1', 'p1', 'writes'), ('a1', 'p2', 'writes'),
            ('a2', 'p2', 'writes'), ('a2', 'p3', 'writes'),
            ('a3', 'p3', 'writes'), ('a3', 'p4', 'writes'),
            ('a4', 'p4', 'writes'), ('a4', 'p5', 'writes'),

            #反向关系
            ('p1', 'a1', 'be_writen'), ('p2', 'a1', 'be_writen'),
            ('p2', 'a2', 'be_writen'), ('p3', 'a2', 'be_writen'),
            ('p3', 'a3', 'be_writen'), ('p4', 'a3', 'be_writen'),
            ('p4', 'a4', 'be_writen'), ('p5', 'a4', 'be_writen'),
            
            # 论文-会议关系 (发表)
            ('p1', 'v1', 'published_in'), ('p2', 'v1', 'published_in'),
            ('p3', 'v2', 'published_in'), ('p4', 'v2', 'published_in'),
            ('p5', 'v1', 'published_in'),

            # 反向论文-会议关系 (发表)
            ('v1', 'p1', 'publish'), ('v1', 'p2', 'publish'),
            ('v2', 'p3', 'publish'), ('v2', 'p4', 'publish'),
            ('v1', 'p5', 'publish'),
            
            # 论文-关键词关系
            ('p1', 'k1', 'has_keyword'), ('p1', 'k2', 'has_keyword'),
            ('p2', 'k1', 'has_keyword'), ('p2', 'k3', 'has_keyword'),
            ('p3', 'k2', 'has_keyword'), ('p3', 'k3', 'has_keyword'),
            ('p4', 'k1', 'has_keyword'), ('p5', 'k2', 'has_keyword'),

             # 反向论文-关键词关系
            ('k1', 'p1', 'keyword_in'), ('k2', 'p1', 'keyword_in'),
            ('k1', 'p2', 'keyword_in'), ('k3', 'p2', 'keyword_in'),
            ('k2', 'p3', 'keyword_in'), ('k3', 'p3', 'keyword_in'),
            ('k1', 'p4', 'keyword_in'), ('k2', 'p5', 'keyword_in'),
            
            # 论文-论文引用关系
            ('p2', 'p1', 'cites'), ('p3', 'p2', 'cites'),
            ('p4', 'p3', 'cites'), ('p5', 'p1', 'cites'),

            # 反向论文-论文引用关系
            ('p1', 'p2', 'cited_in'), ('p2', 'p3', 'cited_in'),
            ('p3', 'p4', 'cited_in'), ('p1', 'p5', 'cited_in')
        ]
    }
    # 定义元路径
    author_meta_paths = [
        'A-P→P-A',       # author citation
        'A-P-A',   # co-author
        'A-P-V-P-A',    # authors with same journal/conference
        'A-P→P←P-A',    # authors’ co-citation
        'A-P←P→P-A',    # author citations
        'A-P-K-P-A'    # authors with same keywords
    ]

    paper_meta_paths = [
        'P→P',       # citation relations
        'P→P←P',   # co-citation
        'P←P→P',    # citations
        'P-A-P',    # with same author
        'P-K-P',    # with same keyword
        'P-V-P'   # with same journal/conference
    ]

    # 预处理（只需执行一次）
    adjacency_list = build_adjacency_list(hetero_graph)
    
    # 以作者为中心的检测
    print("Running author-centric detection...")
    author_communities = HETE_MESE(hetero_graph, 'author', author_meta_paths,adjacency_list)
    author_results = evaluate_communities(hetero_graph, author_communities)
    
    # 以论文为中心的检测
    print("Running paper-centric detection...")
    paper_communities = HETE_MESE(hetero_graph, 'paper', paper_meta_paths,adjacency_list)
    paper_results = evaluate_communities(hetero_graph, paper_communities)
    
    # 打印结果
    print("\nResults:")
    print("Author-centric communities:")
    print(f"- Number of communities: {len(author_communities)}")
    print(f"- Modularity: {author_results['modularity']:.4f}")
    print(f"- Keyword coherence: {author_results['keyword_coherence']:.4f}")
    
    print("\nPaper-centric communities:")
    print(f"- Number of communities: {len(paper_communities)}")
    print(f"- Modularity: {paper_results['modularity']:.4f}")
    print(f"- Keyword coherence: {paper_results['keyword_coherence']:.4f}")
    
    return {
        'author_centric': author_results,
        'paper_centric': paper_results
    }

if __name__ == "__main__":
    results = run_experiments()
    
    # 可视化部分结果
    import matplotlib.pyplot as plt
    
    metrics = ['modularity', 'keyword_coherence']
    values = {
        'Author-centric': [results['author_centric'][m] for m in metrics],
        'Paper-centric': [results['paper_centric'][m] for m in metrics]
    }
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values['Author-centric'], width, label='Author-centric')
    rects2 = ax.bar(x + width/2, values['Paper-centric'], width, label='Paper-centric')
    
    ax.set_ylabel('Scores')
    ax.set_title('Performance by community type')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    fig.tight_layout()
    plt.show()