
import networkx as nx 
from networkx.readwrite import json_graph
import dgl
from collections import Counter, defaultdict
import json
import sys 
import numpy as np
import torch 
from tqdm import tqdm 
import random
import math 
import os 
import config 

def load_data(dataset="flickr", features_key='features'):
    # if dataset == 'flickr':
    # features_size = {'image': 256, 'group': 256, 'user': 256, 'term':256 }
    # features_size = {'image': 512, 'group': 768, 'user': 256, 'term':768 }
    features_size = config.FEATURES_SIZE
    id2entityname = {0:'image', 1:'group',2:'user',3:'term'}
    graph_path = config.GRAPH_PATH
    feats_path = config.FEATS_PATH
    # elif dataset == 'aifb':
    #     id2entityname = {0:'project', 1:'org',2:'person',3:'group', 4:'field',5:'pub',6:'external'}
    #     graph_path = 'data/aifb/aifb-G.json'
    #     feats_path = 'data/aifb/aifb-feats.npy'
    # else:
    #     raise NotImplementedError

    number_of_types = len(id2entityname)

    jsondata = json.load(open(graph_path))
    nx_G = json_graph.node_link_graph(jsondata)

    feats = np.load(feats_path)
    types = defaultdict(list)

    nxid2content = {}
    for node in tqdm(nx_G.nodes(), desc='Read node type'):
        node_type = nx_G.nodes[node]['label'][0]
        assert node_type in range(number_of_types)
        types[node_type].append(node)

        content = nx_G.nodes[node]['content']
        if len(content) == 1:
            content = content[0].lower().replace('"','').replace("'",'').strip()
        elif len(content) == 2:
            content = (content[0], content[1]['url'], content[1]['title'])

        nxid2content[node] = content
    
    node2id = {}
    nx2dgl_map = {}
    for typeid in range(number_of_types):
        node2id[typeid] = {node:i for i, node in enumerate(types[typeid])}
        for i, node in enumerate(types[typeid]):
            nx2dgl_map[node] = (id2entityname[typeid],i)

    content2dglid = {}
    for nx_id, content in nxid2content.items():
        ntype, dgl_id = nx2dgl_map[nx_id]
        content2dglid[(ntype, content)] = dgl_id

    id2node = {}
    dgl2nx_map = {}
    for typeid in range(number_of_types):
        ## i: index of node in dgl graph, node: index of node in nx graph 
        id2node[typeid] = {i:node for i, node in enumerate(types[typeid])}
        for i, node in enumerate(types[typeid]):
            dgl2nx_map[(id2entityname[typeid],i)] = node
    
    id2relaname = {}
    for i in range(number_of_types):
        for j in range(number_of_types):
            if i==j: continue
            id2relaname[(id2entityname[i], id2entityname[j])] = id2entityname[i]+'-->'+id2entityname[j]


    feats_dict = {}
    for i in range(number_of_types):
        name = id2entityname[i]
        feats_dict[name] = np.zeros( (len(id2node[i]), feats.shape[1]) )
        
        for j, node in id2node[i].items():
            feats_dict[name][j] = feats[node]

    edgetype_dict = defaultdict(list)
    for edge in tqdm(nx_G.edges(), desc='Read edge type'):
        node1, node2 = edge
        nodetype1 = nx_G.nodes[node1]['label'][0]
        nodetype2 = nx_G.nodes[node2]['label'][0]
        nodeidx1 = node2id[nodetype1][node1]
        nodeidx2 = node2id[nodetype2][node2]
        
        name1, name2 = id2entityname[nodetype1], id2entityname[nodetype2]
        edgetype_dict[(name1, id2relaname[(name1, name2)], name2)].append((nodeidx1, nodeidx2))
        edgetype_dict[(name2, id2relaname[(name2, name1)], name1)].append((nodeidx2, nodeidx1))
        

    dgl_G = dgl.heterograph(edgetype_dict, num_nodes_dict={id2entityname[i]:len(node2id[i]) for i in range(number_of_types)})
    for nodetype in id2entityname.values():
        dgl_G.nodes[nodetype].data[features_key] = torch.FloatTensor(feats_dict[nodetype][:,:features_size[nodetype]])
    return dgl_G, nx_G, nx2dgl_map, dgl2nx_map, nxid2content, content2dglid, feats, features_size


# class SAGEDataset():
#     def __init__(self,dgl_G, batch_size=512, neg_size=20, swap=False):
#         self.dgl_G = dgl_G
#         self.batch_size = batch_size
#         self.neg_size = neg_size
#         self.swap = swap
#         self.pairs = []
#         self._construct_nx_graph()

#     def _construct_nx_graph(self):
#         dgl2nx_map = {}
#         for ntype in tqdm(self.dgl_G.ntypes, desc='Read nodes from dgl'):
#             for i in range(self.dgl_G.number_of_nodes(ntype)):
#                 dgl2nx_map[(ntype, i)] = len(dgl2nx_map) 

#         nx2dgl_map = {v:k for k,v in dgl2nx_map.items()}

#         nx_graph = nx.Graph()
#         nx_graph.add_nodes_from(list(range(len(nx2dgl_map))))

#         for etype in tqdm(self.dgl_G.canonical_etypes, desc='Read edges from dgl'):
#             ntype_src, _, ntype_dst = etype
#             src_idxs, dst_idxs = self.dgl_G.all_edges(etype=etype)
#             src_idxs = src_idxs.detach().cpu().numpy().tolist()
#             dst_idxs = dst_idxs.detach().cpu().numpy().tolist()

#             for src_idx, dst_idx in zip(src_idxs, dst_idxs):
#                 nx_graph.add_edge(dgl2nx_map[(ntype_src, src_idx)], dgl2nx_map[(ntype_dst, dst_idx)])
        
#         self.nx_G = nx_graph
#         self.iso_nodes = [node for node in nx.isolates(self.nx_G)]
#         self.edges = np.array(self.nx_G.edges)
#         if self.swap:
#             self.edges = np.vstack([self.edges,self.edges[:,::-1]])
#         self.n_edges = self.edges.shape[0]
#         self.n_batches = int(math.ceil(self.n_edges/self.batch_size))

#         self.degrees = np.array([self.nx_G.degree(node) for node in self.nx_G.nodes])   
#         self.max_degree = int(np.max(self.degrees))

#         self.dgl2nx = dgl2nx_map
#         self.nx2dgl = nx2dgl_map

#         ntype_list = [self.nx2dgl[i][0] for i in self.edges[:,0]]
#         ntype_freq = dict(Counter(ntype_list))
#         weights = [ntype_freq[ntype] for ntype in ntype_list]
#         weights = np.array(weights)
#         self.weights = weights/np.sum(weights)

#     def __iter__(self):
#         s = np.arange(self.edges.shape[0])
#         np.random.shuffle(s)
#         self.edges = self.edges[s]
#         self.weights = self.weights[s]

#         batch_idx = 0
#         while(batch_idx<self.n_batches):
#             start = batch_idx*self.batch_size
#             batch_edges = self.edges[start:min(start+self.batch_size, self.n_edges)]
#             batch_weights = self.weights[start:min(start+self.batch_size, self.n_edges)]
#             batch_idx+=1
#             yield batch_edges, batch_weights, self._samples_neg()

#     def _samples_neg(self, unique=False):
#         distortion = 0.75
#         unique = False 

#         weights = self.degrees**distortion
#         prob = weights/weights.sum()
#         sampled = np.random.choice(len(self.degrees), self.neg_size, p=prob, replace=~unique)

#         return sampled

# class QueryMachine():
#     def __init__(self,dgl_G, model, features_key="features"):
#         self.dgl_G = dgl_G
#         self.features_key = features_key
#         self.model = model 
#         self._create_embeddings()
#         self._construct_nx_graph()

#     def _create_embeddings(self):
#         self.embeddings = self.model(self.dgl_G, self.features_key)
#         for k, v in self.embeddings.items():
#             self.embeddings[k] = v.detach().cpu().numpy()

#     def _construct_nx_graph(self):
#         dgl2nx_map = {}
#         for ntype in tqdm(self.dgl_G.ntypes, desc='Read nodes from dgl'):
#             for i in range(self.dgl_G.number_of_nodes(ntype)):
#                 dgl2nx_map[(ntype, i)] = len(dgl2nx_map) 

#         nx2dgl_map = {v:k for k,v in dgl2nx_map.items()}

#         nx_graph = nx.Graph()
#         nx_graph.add_nodes_from(list(range(len(nx2dgl_map))))

#         for etype in tqdm(self.dgl_G.canonical_etypes, desc='Read edges from dgl'):
#             ntype_src, _, ntype_dst = etype
#             src_idxs, dst_idxs = self.dgl_G.all_edges(etype=etype)
#             src_idxs = src_idxs.detach().cpu().numpy().tolist()
#             dst_idxs = dst_idxs.detach().cpu().numpy().tolist()

#             for src_idx, dst_idx in zip(src_idxs, dst_idxs):
#                 nx_graph.add_edge(dgl2nx_map[(ntype_src, src_idx)], dgl2nx_map[(ntype_dst, dst_idx)])
        
#         self.nx_G = nx_graph
#         self.iso_nodes = [node for node in nx.isolates(self.nx_G)]

#         self.dgl2nx = dgl2nx_map
#         self.nx2dgl = nx2dgl_map

#     def _get_neighbors(self, nodeid, ntype="image"):
#         try:
#             nx_id = self.dgl2nx[(ntype, nodeid)]
#         except KeyError:
#             assert "Key not found, maybe nodeid is out of range"
        
#         neighbors = [self.nx2dgl[node] for node in self.nx_G.neighbors(nx_id)]
        
#         return neighbors

#     def _build_dgl(self, nodeid, ntype="image", keep_rate=1):
#         neighbors = self._get_neighbors(nodeid, ntype)
#         if keep_rate <1:
#             neighbors = random.sample(neighbors, int(keep_rate*len(neighbors)))

#         node_dict = defaultdict(list)
#         for neigh_ntype, neigh_id in neighbors:
#             node_dict[neigh_ntype].append(neigh_id)
#         node_dict[ntype].append(nodeid)

#         sub_dgl_G = self.dgl_G.subgraph(node_dict)
#         sub_dgl_G.nodes[ntype].data[self.features_key][-1] = torch.zeros(sub_dgl_G.nodes[ntype].data[self.features_key][-1].size())

#         return sub_dgl_G
    
#     def eval_rank(self, nodeid, ntype="image", keep_rate=1):
#         sub_dgl_G = self._build_dgl(nodeid, ntype, keep_rate)
#         sub_embeddings = self.model(sub_dgl_G, self.features_key)[ntype].detach().cpu().numpy()
#         distances = ((sub_embeddings[-1] - self.embeddings[ntype])**2).sum(1)
#         sorted_idx = np.argsort(distances)
#         for i, idx in enumerate(sorted_idx):
#             if idx == nodeid:
#                 return i 

            
# if __name__ == "__main__":
#     import pdb
#     dgl_G, nx_G, nx2dgl_map, dgl2nx_map, id2content, content2id = load_data()
#     pdb.set_trace()
