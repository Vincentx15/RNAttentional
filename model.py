import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RGATLayer


class RGATEmbedder(nn.Module):
    """
    This is an exemple RGCN for unsupervised learning, going from one element of "dims" to the other

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 dims,
                 num_heads=3,
                 sample_other=0.2,
                 infeatures_dim=0,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 return_loss=True,
                 verbose=False):
        super(RGATEmbedder, self).__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.use_node_features = (infeatures_dim != 0)
        self.in_dim = 1 if infeatures_dim == 0 else infeatures_dim
        self.conv_output = conv_output
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.verbose = verbose
        self.return_loss = return_loss

        self.layers = self.build_model()

        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = self.dims[-2:]
        if self.verbose:
            print("short, ", short)
            print("last_hidden, last ", last_hidden, last)

        # input feature is just node degree
        i2h = RGATLayer(in_feat=self.in_dim,
                        out_feat=self.dims[0],
                        num_rels=self.num_rels,
                        num_bases=self.num_bases,
                        num_heads=self.num_heads,
                        sample_other=self.sample_other,
                        activation=F.relu,
                        self_loop=self.self_loop)
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                            out_feat=dim_out,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            layers.append(h2h)

        # hidden to output
        if self.conv_output:
            h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                            out_feat=last,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            self_loop=self.self_loop,
                            activation=None)
        else:
            h2o = nn.Linear(last_hidden * self.num_heads, last)
        layers.append(h2o)
        return layers

    def deactivate_loss(self):
        for layer in self.layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g):
        iso_loss = 0
        if self.use_node_features:
            h = g.ndata['features']
        else:
            # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            if not self.conv_output and (i == len(self.layers) - 1):
                h = layer(h)
            else:
                if layer.return_loss:
                    h, loss = layer(g=g, feat=h)
                    iso_loss += loss
                else:
                    h = layer(g=g, feat=h)
        if self.return_loss:
            return h, iso_loss
        else:
            return h


class RGATClassifier(nn.Module):
    """
    This is an exemple RGCN for supervised learning, that uses the previous Embedder network

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 rgat_embedder,
                 classif_dims=None,
                 num_heads=5,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 verbose=False,
                 return_loss=True,
                 sample_other=0.2):
        super(RGATClassifier, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.conv_output = conv_output
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.return_loss = return_loss

        self.rgat_embedder = rgat_embedder
        self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads
        self.classif_dims = classif_dims

        self.classif_layers = self.build_model()

        self.verbose = verbose
        if self.verbose:
            print(self.classif_layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        if self.classif_dims is None:
            return self.rgat_embedder

        classif_layers = nn.ModuleList()
        # Just one convolution
        if len(self.classif_dims) == 1:
            if self.conv_output:
                h2o = RGATLayer(in_feat=self.last_dim_embedder,
                                out_feat=self.classif_dims[0],
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                # Old fix for a bug in dgl<0.6
                                # self_loop=self.self_loop and self.classif_dims[0] > 1,
                                activation=None)
            else:
                h2o = nn.Linear(self.last_dim_embedder, self.classif_dims[0])
            classif_layers.append(h2o)
            return classif_layers

        # The supervised is more than one layer
        else:
            i2h = RGATLayer(in_feat=self.last_dim_embedder,
                            out_feat=self.classif_dims[0],
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            classif_layers.append(i2h)
            last_hidden, last = self.classif_dims[-2:]
            short = self.classif_dims[:-1]
            for dim_in, dim_out in zip(short, short[1:]):
                h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                                out_feat=dim_out,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                activation=F.relu,
                                self_loop=self.self_loop)
                classif_layers.append(h2h)

            # hidden to output
            if self.conv_output:
                h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                                out_feat=last,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                activation=None)
            else:
                h2o = nn.Linear(last_hidden * self.num_heads, last)
            classif_layers.append(h2o)
            return classif_layers

    def deactivate_loss(self):
        self.return_loss = False
        self.rgat_embedder.deactivate_loss()
        for layer in self.classif_layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g):
        iso_loss = 0
        if self.rgat_embedder.return_loss:
            h, loss = self.rgat_embedder(g)
        else:
            h = self.rgat_embedder(g)
            loss = 0
        iso_loss += loss
        for i, layer in enumerate(self.classif_layers):
            # if this is the last layer and we want to use a linear layer, the call is different
            if (i == len(self.classif_layers) - 1) and not self.conv_output:
                h = layer(h)
            # Convolution layer
            else:
                if layer.return_loss:
                    h, loss = layer(g, h)
                    iso_loss += loss
                else:
                    h = layer(g, h)

        if self.return_loss:
            return h, iso_loss
        else:
            return h
