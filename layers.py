import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels,
                 num_heads=5,
                 self_interaction=True,
                 num_bases=-1,
                 iso_mat=None,
                 sample_other=0.):
        """

        :param in_dim:
        :param out_dim:
        :param num_rels:
        :param num_heads:
        :param self_interaction:
        :param num_bases:
        :param iso_mat:
        :param sample_other: The frequency with which we change the edges in the reconstruction loss
        """
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
        self.attention_weight = nn.Parameter(torch.Tensor(self.num_bases, 2 * out_dim, num_heads))
        if self.use_basis_sharing:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # This is for the sample other regularisation ie isostericity
        self.sample_other = sample_other
        if sample_other > 0 and iso_mat is None:
            raise ValueError('Cannot use isostericity regularisation if iso_mat is not provided')
        if isinstance(iso_mat, np.ndarray):
            self.iso_mat = torch.from_numpy(iso_mat)

        # self.attn_fcs = nn.Parameter(torch.Tensor(self.num_rels, 2 * out_dim, num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        if self.self_interaction:
            nn.init.xavier_normal_(self.self_fc.weight, gain=gain)

        # nn.init.xavier_normal_(self.attn_fcs, gain=gain)
        nn.init.xavier_normal_(self.attention_weight, gain=gain)
        if self.use_basis_sharing:
            nn.init.xavier_normal_(self.w_comp, gain=gain)

    def edge_attention(self, edges):
        """
        This is where the relational part hits

        Warning : this needs to stay updated with its sampled copy (could be a factory function
        """
        # First build the correct attention scheme if using basis sharing
        if self.use_basis_sharing:
            attn_fcs = torch.einsum('ab,bcd->acd', self.w_comp, self.attention_weight)
        else:
            attn_fcs = self.attention_weight

        # Then apply the right Tensor to each concatenated message based on edge type
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        edge_types = edges.data['rel_type']
        w = attn_fcs[edge_types]
        a = torch.bmm(z2.unsqueeze(1), w)
        a = a.squeeze()
        # Todo : think about using no activation
        return {'attention': F.leaky_relu(a)}

    def sampled_edge_attention(self, edges):
        """
        We need a copy of the function as it causes problem to call a function with arguments
        in the special dgl functions
        """
        # First build the correct attention scheme if using basis sharing
        if self.use_basis_sharing:
            attn_fcs = torch.einsum('ab,bcd->acd', self.w_comp, self.attention_weight)
        else:
            attn_fcs = self.attention_weight

        # Then apply the right Tensor to each concatenated message based on edge type
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)

        # We then alter a fraction of the edges and
        # compute the message that would be passed over this pertubed adjacency
        # Todo : think about different sampling schemes
        edge_types = edges.data['rel_type']
        edge_types_perturbation = torch.randint(0, self.num_rels, size=edges.data['rel_type'].shape)
        sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([self.sample_other]))
        binary_mask = sampler.sample(edge_types.shape).type(torch.bool).squeeze()
        modified_edge_types = edge_types + edge_types_perturbation * binary_mask
        modified_edge_types = torch.remainder(modified_edge_types, torch.tensor(self.num_rels))
        w = attn_fcs[modified_edge_types]
        sampled_attention = torch.bmm(z2.unsqueeze(1), w)
        sampled_attention = sampled_attention.squeeze()
        return {'sampled_attention': F.leaky_relu(sampled_attention), 'modified_edge_types': modified_edge_types}

    def compute_isostericity_loss(self, g):
        """
        Compare the attentions induced distances with the isostericity matrix
        :param g:
        :return:
        """
        native_attentions = g.edata['attention']
        modified_attentions = g.edata['sampled_attention']
        edge_types = g.edata['rel_type']
        modified_edge_types = g.edata['modified_edge_types']

        similarities = self.iso_mat[edge_types, modified_edge_types]
        computed_distances = torch.nn.PairwiseDistance()(native_attentions, modified_attentions)
        computed_similarities = torch.exp(-computed_distances)
        isostericity_loss = nn.MSELoss()(similarities, computed_similarities)

        # isostericity_loss = torch.sum(similarities * computed_distances)
        return isostericity_loss

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
                                       all_attentions, all_messages)  # shape : (similar_nodes, nei, heads, out_dim)
        concatenated = torch.flatten(scaled_messages, start_dim=2)  # shape : (similar_nodes, nei, heads * out_dim)
        h = torch.sum(concatenated, dim=1)  # shape : (similar_nodes, heads * out_dim)

        if self.self_interaction:
            self_z = self.self_fc(nodes.data['h'])
            h = h + self_z
        return {'h': h}

    def forward(self, g):
        h = g.ndata['h']
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)

        sample_loss = None
        if self.sample_other > 0.:
            g.apply_edges(self.sampled_edge_attention)
            sample_loss = self.compute_isostericity_loss(g)

        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h'), sample_loss
