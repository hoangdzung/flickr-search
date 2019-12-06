from collections import defaultdict
from utils import load_data
from model import HeteroRGCN
import torch 
import dgl 
import networkx as nx
import numpy as np 

features_key = "features"
dgl_G, nx_G, nx2dgl_map, dgl2nx_map, id2content, content2id, feats = load_data(features_key=features_key)
isolate_nodes = set([node for node in nx.isolates(nx_G)])

print("Load model...")
model = HeteroRGCN(dgl_G, 256, 64, 32)

#try:
model.load_state_dict(torch.load('model/firts_norm/model.pt'))
if torch.cuda.is_availabel():
    model = model.cuda()
#except:
#    print("Can't load mode and use cuda")

model.eval()

embeddings = model(dgl_G)
for ntype in embeddings:
    embeddings[ntype] = embeddings[ntype].detach().cpu().numpy()
    print(ntype, embeddings[ntype].shape)

def process(results):    
    query_type, neigh_dgl_idxs = parse_data(results)
    print("List of indexes in dgl Graph:", neigh_dgl_idxs)
    if (len(neigh_dgl_idxs)) == 0:
        return [('static/images/notfound.png', '/', 'Term not found')]
    query_dgl_G = build_query_dgl_graph(query_type, neigh_dgl_idxs)

    query_embedding = model(query_dgl_G)[query_type][0].detach().cpu().numpy()
    
    candidates_nx_id = get_candidates(query_embedding, query_type)

    contents = [id2content[i] for i in candidates_nx_id]

    return [('static/images/regular/{}.jpg'.format(content[0]), content[1], content[2]) for content in contents]

def get_candidates(query_embedding, query_type):
    candidate_embeddings = embeddings[query_type]
    distances = ((query_embedding-candidate_embeddings)**2).sum(1)
    candidates_nx_id = []
    for dgl_id in np.argsort(distances):
        nx_idx = dgl2nx_map[(query_type, dgl_id)]
        if nx_idx not in isolate_nodes:
            candidates_nx_id.append(nx_idx)
        if len(candidates_nx_id) ==10:
            break
    return candidates_nx_id

def parse_data(results):
    ### default ntype is image
    query_type = "image"
    keywords = defaultdict(list)
    for key in results:
        if key == "category":
            query_type = results["category"]
        else:
            keywords[key] = [i.replace('"','').replace("'",'').strip() for i in results[key].lower().split(";")]

    neigh_dgl_idxs = defaultdict(list)
    for ntype, contents in keywords.items():
        for content in contents:
            nx_idx = content2nxid(ntype, content)
            if nx_idx>=0:
                neigh_dgl_idxs[ntype].append(nx_idx)

    return query_type, neigh_dgl_idxs

def content2nxid(ntype, content):
    try:
        nx_idx = content2id[content]
    except KeyError:
        return -1 
    else:
        dgl_idx = nx2dgl_map[nx_idx]
        if dgl_idx[0] == ntype:
            return nx_idx
        else:
            return -1

def build_query_dgl_graph(query_type, neigh_dgl_idxs):
    edges_dict = defaultdict(list)
    feats_dict = defaultdict(list)
    for neigh_ntype in neigh_dgl_idxs:
        for i, neigh_id in enumerate(neigh_dgl_idxs[neigh_ntype]):
            edges_dict[(query_type, query_type+'-->'+neigh_ntype, neigh_ntype)].append((0,i))
            edges_dict[(neigh_ntype, neigh_ntype+'-->'+query_type, query_type)].append((i,0))
            feats_dict[neigh_ntype].append(feats[neigh_id])
    
    feats_dict[query_type] = [np.zeros(feats.shape[-1])]
    query_dgl_G = dgl.heterograph(edges_dict)

    for ntype in query_dgl_G.ntypes:
        query_dgl_G.nodes[ntype].data[features_key] = torch.FloatTensor(np.stack(feats_dict[ntype]))

    return query_dgl_G
