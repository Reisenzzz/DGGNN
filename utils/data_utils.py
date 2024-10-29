import numpy as np
import numpy as np
import torch

from torch.utils.data import Dataset


def get_adj(maxdistance:int = 200):
    distance = np.load('./data/pm2.5/station_distance.npz')['arr_0']
    d = (maxdistance-distance)/maxdistance
    d = np.where(d>0,d,0)
    adj = np.where(d>0,1,0)
    adj = torch.from_numpy(adj).float()
    d = torch.from_numpy(d).float()
    return adj,d

def get_adj2(maxdistance:int = 200):
    distance = np.load('./data/pm2.5/station_distance.npz')['arr_0']
    d = np.where(distance>1,distance,1)
    d = np.where(d<maxdistance,1/d,0)
    adj = np.where(distance<maxdistance,1,0)
    adj = torch.from_numpy(adj).float()
    d = torch.from_numpy(d).float()
    return adj,d

class PollutionDataset(Dataset):

    def __init__(self,data,graph,n_his:int=12,n_pred:int=36) -> None:
        self.n_his = n_his
        self.n_pred = n_pred
        self.data = data
        self.graph = graph

    def __getitem__(self, index):
        data_x = self.data[:,index:(index+self.n_his),:]
        data_y = self.data[:,(index+self.n_his):(index+self.n_his+self.n_pred),0]   
        data_graphs = self.graph[:,:,index:(index+self.n_his)]

        return data_x,data_y,data_graphs

    def __len__(self):
        return (self.data.size(1)-self.n_his-self.n_pred)