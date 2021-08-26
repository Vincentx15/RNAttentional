import dgl
import torch
import networkx as nx

from graph_utils import load_json
from iso_mat import ISO_MAT, EDGE_MAP
from layers import RGATLayer
from model import RGATClassifier, RGATEmbedder

if __name__ == '__main__':
    # Create the relattentional layer
    in_dim = 3
    out_dim = 4
    num_rels = len(EDGE_MAP)
    rgat = RGATLayer(in_feat=in_dim, out_feat=out_dim, num_rels=num_rels, num_bases=3, sample_other=0.5,
                     iso_mat=ISO_MAT)

    # Get graph, add constant ones on the first layer, one hot edge encoding RGCN runs and send to DGL
    graph_path = '5e3h.json'
    graph = load_json(graph_path)
    constant_nodes = {node: torch.ones(in_dim) for node in graph.nodes()}
    nx.set_node_attributes(graph, name='features', values=constant_nodes)
    one_hot = {edge: torch.tensor(EDGE_MAP[label]) for edge, label in
               (nx.get_edge_attributes(graph, 'LW')).items()}
    nx.set_edge_attributes(graph, name='edge_type', values=one_hot)
    g_dgl = dgl.from_networkx(nx_graph=graph, node_attrs=['features'], edge_attrs=['edge_type'])

    # Make a forward call
    feat = g_dgl.ndata['features']
    h, iso_loss = rgat(g_dgl, feat=feat)
    print(iso_loss)

    rgat_embedder = RGATEmbedder(dims=[10, 10])
    rgat_model = RGATClassifier(rgat_embedder, classif_dims=[10, 10], verbose=True)
    h, iso_loss = rgat_model(g_dgl)
    print(iso_loss)
