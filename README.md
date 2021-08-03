# RNAttentional

This project is meant to introduce a new graph Layer, that takes as input an 
isostericity matrix. This matrix denotes a similarity between edges and is used in 
the RNA community.

## Mathematical framing

The idea is quite simple, we do a simple RGAT update, but we add a control over
how messages differ when they are sent through different channels.
To do so in a simple way, we control the euclidean distance between attention weights
and force them to mimick the similarity values.

### TODO : extension 
We should maybe downscale in the loss, the attentions that are close to zero 
for different messages as they might just say that a message is not interesting 
regardless of the channel.

## Implementation
The project is very minimal, one should instantiate the layer with an isostericity
matrix whose order follows the one of the edge encoding of graphs.
Then one can just use the graph as usual and one gets an extra return 
from the forward call that represents the isostericity loss.


