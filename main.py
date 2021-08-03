import dgl
import torch
import networkx as nx

from graph_utils import load_json
from iso_mat import ISO_MAT, EDGE_MAP
from layers import RGATLayer

if __name__ == '__main__':

    # Create the relattentional layer
    in_dim = 3
    out_dim = 4
    num_rels = len(EDGE_MAP)
    rgat = RGATLayer(in_dim=in_dim, out_dim=out_dim, num_rels=num_rels, num_bases=3, sample_other=0.5, iso_mat=ISO_MAT)

    # Get graph, add constant ones on the first layer, one hot edge encoding RGCN runs and send to DGL
    graph_path = '5e3h.json'
    graph = load_json(graph_path)
    constant_nodes = {node: torch.ones(in_dim) for node in graph.nodes()}
    nx.set_node_attributes(graph, name='h', values=constant_nodes)
    one_hot = {edge: torch.tensor(EDGE_MAP[label]) for edge, label in
               (nx.get_edge_attributes(graph, 'LW')).items()}
    nx.set_edge_attributes(graph, name='rel_type', values=one_hot)
    g_dgl = dgl.from_networkx(nx_graph=graph, node_attrs=['h'], edge_attrs=['rel_type'])

    # Make a forward call
    rgat(g_dgl)
