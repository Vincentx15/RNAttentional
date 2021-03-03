import dgl


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial


class RGATLayer(nn.module):
    def __init__(self, g, in_dim, out_dim, num_rels, num_heads=5):
        super(RGATLayer, self).__init__()
        self.g = g
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        # Todo : try the basis sharing trick
        self.attn_fcs = [nn.Linear(2 * out_dim, num_heads, bias=False) for _ in range(num_rels)]
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        for attn_fc in self.attn_fcs:
            nn.init.xavier_normal_(attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        """
        This is where the relational part hits
        """
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        net_to_use = self.attn_fcs[edges.data['rel_type']]
        a = net_to_use(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """
        Now use concatenation for the aggregation
        """
        all_attentions = nodes.mailbox['e']   # shape : (nei, heads)
        all_messages = nodes.mailbox['z']   # shape : (nei, out_dim)
        all_attentions.unsqueeze(-1)
        all_messages.unsqueeze(-2)
        concats = torch.bmm(all_attentions, all_messages)
        h = torch.sum(concats, dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


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







