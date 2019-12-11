#MODEL_PATH = '../flickr-dgl/model/swap_flickr_768/model.pt'
MODEL_PATH = '../flickr-dgl/model/swap_att_flickr_768/model.pt'
#FEATURES_SIZE = {'image': 256, 'group': 256, 'user': 256, 'term':256 }
FEATURES_SIZE = {'image': 512, 'group': 768, 'user': 256, 'term':768 }
GRAPH_PATH = 'data/flickr/flickr-G-formatted.json'
FEATS_PATH = 'data/flickr/flickr-new-feats_768.npy'
