import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

import networkx as nx


class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_heads=5, self_interaction=True, num_bases=-1):
        super(RGATLayer, self).__init__()
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.self_interaction = self_interaction
        if self_interaction:
            self.self_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        # Basis sharing trick
        if num_bases <= 0 or num_bases > self.num_rels:
            self.num_bases = num_rels
            self.use_basis_sharing = False
        else:
            self.num_bases = num_bases
            self.use_basis_sharing = True

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, 2 * out_dim, num_heads))
        if self.use_basis_sharing:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # self.attn_fcs = nn.Parameter(torch.Tensor(self.num_rels, 2 * out_dim, num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        if self.self_interaction:
            nn.init.xavier_normal_(self.self_fc.weight, gain=gain)

        # nn.init.xavier_normal_(self.attn_fcs, gain=gain)
        nn.init.xavier_normal_(self.weight, gain=gain)
        if self.use_basis_sharing:
            nn.init.xavier_normal_(self.w_comp, gain=gain)

    def edge_attention(self, edges):
        """
        This is where the relational part hits
        """
        # First build the correct attention scheme if using basis sharing
        if self.use_basis_sharing:
            attn_fcs = torch.einsum('ab,bcd->acd', (self.w_comp, self.weight))
        else:
            attn_fcs = self.weight

        # Then apply the right Tensor to each concatenated message based on edge type
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        w = attn_fcs[edges.data['rel_type']]
        a = torch.bmm(z2.unsqueeze(1), w)
        a = a.squeeze()
        # Todo : think about using no activation
        return {'attention': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'attention': edges.data['attention']}

    def reduce_func(self, nodes):
        """
        Now use concatenation for the aggregation
        """
        all_attentions = nodes.mailbox['attention']  # shape : (similar_nodes, nei, heads)
        all_messages = nodes.mailbox['z']  # shape : (similar_nodes, nei, out_dim)

        # Compute scaling of messages with einsum and concatenate, then aggregate over neighbors
        scaled_messages = torch.einsum('abi,abj->abij',
                                       (all_attentions, all_messages))  # shape : (similar_nodes, nei, heads, out_dim)
        concatenated = torch.flatten(scaled_messages, start_dim=2)  # shape : (similar_nodes, nei, heads * out_dim)
        h = torch.sum(concatenated, dim=1)  # shape : (similar_nodes, heads * out_dim)

        if self.self_interaction:
            self_z = self.self_fc(nodes.data['h'])
            h = h + self_z
        # print(all_attentions.shape)
        # print(all_messages.shape)
        # print(scaled_messages.shape)
        # # print(concatenated.shape)
        # print(self_z.shape)
        # print(h.shape)
        # print()
        return {'h': h}

    def forward(self, g):
        h = g.ndata['h']
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                self.out_feat))
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


import os
import json
from networkx.readwrite import json_graph


def dump_json(filename, graph):
    g_json = json_graph.node_link_data(graph)
    json.dump(g_json, open(filename, 'w'), indent=2)


def load_json(filename):
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph


if __name__ == '__main__':
    graph_path = '5e3h.json'

    edge_map = {'B53': 0, 'cHH': 1, 'cHS': 2, 'cHW': 3, 'cSH': 2, 'cSS': 4, 'cSW': 5, 'cWH': 3, 'cWS': 5, 'cWW': 6,
                'tHH': 7, 'tHS': 8, 'tHW': 9, 'tSH': 8, 'tSS': 10, 'tSW': 11, 'tWH': 9, 'tWS': 11, 'tWW': 12}

    in_dim = 3
    out_dim = 4
    num_rels = len(edge_map)

    graph = load_json(graph_path)
    graph = nx.to_undirected(graph)
    graph = nx.to_directed(graph)

    # Add constant ones on the first layer
    constant_nodes = {node: torch.ones(in_dim) for node in graph.nodes()}
    nx.set_node_attributes(graph, name='h', values=constant_nodes)

    # Add one hot edge encoding and fake norm fur RGCN runs
    one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in
               (nx.get_edge_attributes(graph, 'LW')).items()}
    fake_norm = {edge: torch.ones(1) for edge in graph.edges()}
    nx.set_edge_attributes(graph, name='norm', values=fake_norm)
    nx.set_edge_attributes(graph, name='rel_type', values=one_hot)

    # Send to DGL
    g_dgl = dgl.from_networkx(nx_graph=graph, node_attrs=['h'], edge_attrs=['rel_type', 'norm'])

    # rgcn = RGCNLayer(in_feat=in_dim, out_feat=out_dim, num_rels=num_rels)
    # rgcn(g_dgl)
    #
    rgat = RGATLayer(in_dim=in_dim, out_dim=out_dim, num_rels=num_rels, num_bases=3)
    rgat(g_dgl)
