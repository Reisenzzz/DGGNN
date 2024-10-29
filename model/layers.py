import torch
from torch import Tensor
from torch.nn import Parameter

from typing import Optional

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
)

from torch_geometric.utils import spmm

def random_walk_normalize(adjacency_matrix):
    degree = torch.sum(adjacency_matrix, dim=1) + 1e-6
    degree_inv = 1.0 / degree
    degree_inv[degree_inv == float('inf')] = 0
    degree_inv_diag = torch.diag(degree_inv)
    adjacency_normalized = torch.mm(degree_inv_diag, adjacency_matrix)
    
    return adjacency_normalized

class DGGConv(MessagePassing):
    def __init__(
        self,
        n_his: int,
        n_pred: int,
        node_num:int,
        self_loop : float = 0.5,
        bias:bool = True,
        d_matrix:Optional[torch.dtype] = None,
        
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.node_num = node_num
        self.in_channels = n_his
        self.out_channels = n_pred
        self.self_loop = self_loop
        self.d = d_matrix
        self.Mask = Parameter(torch.empty(node_num,node_num,self.in_channels))
        loop = torch.diag(torch.ones(self.node_num)).cuda()
        self.loop = loop * self.self_loop
        self.lin = Linear(self.in_channels,self.in_channels,bias=False,weight_initializer='glorot')
        if not(self.d.all() == None):
            self.adj = self.d.cuda()
            u = torch.where(self.adj>0,1,0)
            mask = u.unsqueeze(-1) * torch.ones(self.node_num,self.node_num,self.in_channels).cuda()
            mask = mask.unsqueeze(0)
            self.Mask = Parameter(mask)
        if bias:
            self.bias = Parameter(torch.empty(self.in_channels)).cuda()

        self.idx = torch.nonzero(self.adj).T
        self.adj = self.adj.unsqueeze(0)
        self.loop = self.loop.unsqueeze(0)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, adj_matrix:Tensor) -> Tensor:

        batch_size = x.size(0)
        
        x = self.lin(x)

        out = torch.empty([batch_size,self.node_num,self.in_channels]).cuda()

        for channels in range(self.in_channels):
            adj = adj_matrix[:,:,:,channels]
            adj = adj * self.Mask[:,:,:,channels] * self.adj
            adj = adj + self.loop
            for batch in range(batch_size):
                adj_r = adj[batch,:,:] 
                adj_r = random_walk_normalize(adj_r)
                value = adj_r[self.idx[0],self.idx[1]]  
                x_r = x[batch,:,channels].reshape([self.node_num,1])
                out_r = self.propagate(self.idx,x = x_r,edge_weight= value)
                out[batch,:,channels] = out_r.squeeze(1)
        if self.bias is not None:
            out = out+self.bias            

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)