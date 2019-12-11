import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteroRGCNLayer(nn.Module):
    def __init__(self, G, out_size):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                # name : nn.Linear(in_size, out_size) for name in etypes
                etype : nn.Linear(G.nodes[srctype].data["features"].shape[-1], out_size) for srctype, etype, dsttype in G.canonical_etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroAttRGCNLayer(nn.Module):
    def __init__(self, G, out_size):
        super(HeteroAttRGCNLayer, self).__init__()
        # W_r for each relation
        self.fc = nn.ModuleDict({
                # name : nn.Linear(in_size, out_size) for name in etypes
                etype : nn.Linear(G.nodes[srctype].data["features"].shape[-1], out_size) for srctype, etype, dsttype in G.canonical_etypes
            })
        self.attn_fc = nn.Linear(2 * out_size, 1, bias=False)

    def edge_attention(self, edges, etype):
        srctype, etype, dsttype = etype
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['Wh_%s' % etype],\
                        edges.dst['Wh_%s' % dsttype]], dim=1)
        a = self.attn_fc(z2)
        return {'e_%s' % etype: F.leaky_relu(a)}
    
    def message_func(self, edges, etype):
        # message UDF for equation (3) & (4)
        return {'Wsrc_%s' % (etype): edges.src['Wh_%s' % (etype)],\
                'e_%s' % etype: edges.data['e_%s' % etype]}

    def reduce_func(self, nodes, etype):
        srctype, etype, dsttype = etype

        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e_%s' % etype], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['Wsrc_%s' % (etype)], dim=1)
        return {'self_%s' %(srctype): h}

    def update_nodes(self, nodes):
        feats = []
        for k in nodes.data.keys():
            if k.startswith("self"):
                feats.append(nodes.data[k])
        return {'h':torch.stack(feats).sum(0)}
    
    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.fc[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
  
        for srctype, etype, dsttype in G.canonical_etypes:
            # Cal attention node pair-wise
            G.apply_edges(lambda edges: self.edge_attention(edges,(srctype, etype, dsttype)), etype=etype)

        for srctype, etype, dsttype in G.canonical_etypes:
            G[etype].update_all(lambda edges: self.message_func(edges, etype), \
                                lambda nodes: self.reduce_func(nodes, (srctype, etype, dsttype)), etype=etype)

        for ntype in G.ntypes:
            g.apply_nodes(lambda nodes: self.update_nodes(nodes), ntype=ntype)

        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size = 256, hidden_size=128, out_size=64, learn_feats=True, attention=False):
        super(HeteroRGCN, self).__init__()
        # self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        # self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)
        self.learn_feats = learn_feats
        if self.learn_feats:
            self.user_feats = nn.Parameter(torch.Tensor(G.number_of_nodes("user"), in_size), requires_grad=True)
            nn.init.xavier_uniform_(self.user_feats)
        if attention:
            self.layer = HeteroAttRGCNLayer(G, out_size)
        else:
            self.layer = HeteroRGCNLayer(G, out_size)

    def get_user_feats(self, indexs):
        return self.user_feats[indexs].detach().cpu().numpy()

    def forward(self, G, features_key="features", learn_feats=True):
        embed = nn.ParameterDict({ntype : nn.Parameter(G.nodes[ntype].data[features_key], requires_grad=False) 
                        if  not (ntype == "user" and learn_feats and self.learn_feats) else self.user_feats
                          for ntype in G.ntypes})

        if next(self.parameters()).is_cuda:
            embed =embed.cuda()
        # h_dict = self.layer1(G, embed)
        # h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        # h_dict = self.layer2(G, h_dict)
        h_dict = self.layer(G, embed)
        # get paper logits
        return h_dict
