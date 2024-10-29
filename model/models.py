from model.layers import DGGConv
import torch

    
class DGGNN(torch.nn.Module):
    def __init__(
            self,
            args,
            node_num:int,
            d_matrix:torch.Tensor = None,
    ):
        super(DGGNN,self).__init__()

        self.in_channels = args.n_his
        self.out_channels = args.n_pred
        self.node_num = node_num
        self.MConv = DGGConv(
                n_his=self.in_channels,
                n_pred=self.out_channels,
                node_num=node_num,
                self_loop=args.self_loop,
                bias = args.enable_bias,
                d_matrix=d_matrix,
            )
        self.lin = torch.nn.Linear(args.features,1)
        self.rnn = torch.nn.LSTM(input_size=node_num,hidden_size=args.hidden_size,num_layers=args.num_layer,bias=args.enable_bias,batch_first=True)
        self.fc = torch.nn.Linear(args.hidden_size,self.out_channels * node_num)

    def forward(
        self,
        x:torch.FloatTensor,
        adj_matrix:torch.Tensor
    ) -> torch.FloatTensor:
        batch = x.size(0)
        x = self.lin(x)
        x = x.squeeze(-1)
        x = self.MConv(x,adj_matrix)
        x = x.permute(0,2,1)
        x,_  = self.rnn(x)
        output = x[:,-1,:]
        pred = self.fc(output).view(batch,self.node_num,self.out_channels)
        
        return pred