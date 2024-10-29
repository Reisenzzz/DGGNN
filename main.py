import numpy as np
import numpy as np
import torch
import math
from torch import optim,nn
import logging
import os
from torch.utils.data import DataLoader,random_split

import argparse
import random
from sklearn import preprocessing

from model import models
from tqdm import tqdm

from utils.data_utils import PollutionDataset, get_adj2
from utils import earlystopping


def set_env(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parameters():
    parser = argparse.ArgumentParser(description='DGGNN')
    parser.add_argument('--model_name',type=str,default='DGGNN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=10000, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--x_data_path',type=str,default='./data/pm2.5/data.npy')
    parser.add_argument('--dynamic_graph_data_path',type=str,default='./data/pm2.5/wind.npy')
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=36, help='the number of time interval for predcition, default as 36')
    parser.add_argument('--node_num', type=int, default=50, help='the number of nodes, default as 50')
    parser.add_argument('--self_loop', type=float, default=0.5, help='the self_loop of graph convolution, default as 0.5')
    parser.add_argument('--features', type=int, default=15, help='the number of features, default as 15')
    parser.add_argument('--hidden_size', type=int, default=128, help='the hidden size of LSTM, default as 128')
    parser.add_argument('--num_layer', type=int, default=2, help='the num_layer of LSTM, default as 128')
    parser.add_argument('--val_and_test_rate', type=float, default=0.15, help='the rate of validation and test dataset, default as 0.15')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    return args, device

def data_prepare(args,device):
    data = np.load(args.x_data_path)
    graphs = np.load(args.dynamic_graph_data_path)
    main_scaler = preprocessing.MinMaxScaler()
    other_scaler = preprocessing.MinMaxScaler()

    shape = data.shape
    data = data.reshape(-1,data.shape[2])
    data_main_feature= data[:,0]
    data_other_feature = data[:,1:]
    data_main_feature = main_scaler.fit_transform(data_main_feature.reshape(-1,1))
    data_other_feature  = other_scaler.fit_transform(data_other_feature)
    data = np.concatenate((data_main_feature,data_other_feature),axis=1)
    data = data.reshape(shape)
    

    data = torch.from_numpy(data).float()
    graphs = torch.from_numpy(graphs).float()
    data = data.to(device)
    graphs = graphs.to(device)
    _,d = get_adj2()
    d = d.to(device)
    node_num = d.size(-1)
    val_and_test_rate = 0.15

    total_dataset = PollutionDataset(data=data,graph=graphs,n_his=args.n_his,n_pred=args.n_pred)

    total_len = total_dataset.__len__()
    len_val = int(math.floor(total_len * val_and_test_rate))
    len_test = int(math.floor(total_len * val_and_test_rate))
    len_train = int(total_len - len_val - len_test)
    train_dataset,val_dataset,test_dataset = random_split(total_dataset,[len_train,len_val,len_test])

    train_dataloader = DataLoader(dataset=train_dataset, shuffle = True, batch_size=args.batch_size)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=args.batch_size)

    return main_scaler, d, node_num, train_dataloader, val_dataloader, test_dataloader

def prepare_model(args,d,node_num,device):
    loss = nn.MSELoss()
    model = models.DGGNN(args,node_num,d).to(device)
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')
    
    return loss,model,optimizer,es

def train(loss, args, optimizer, model, train_iter, val_iter,es):
    best_loss = np.inf
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x,y,graphs in tqdm(train_iter):
            y_pred = model(x,graphs)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val(model, val_iter, loss)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        log = 'Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc)
        logger.info(log)
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        if(val_loss < best_loss):
            best_loss = val_loss
            modeldir = './model_params/' + str(args.model_name)
            if os.path.isdir(modeldir) == False :
                os.mkdir(modeldir)
            modelpath = modeldir + '/model.pt'
            torch.save(model,modelpath)

        if es.step(val_loss):
            print('Early stopping.')
            break

@torch.no_grad()
def val(model, val_iter, loss):
    model.eval()
    l_sum, n = 0.0, 0
    for x,y,graphs in val_iter:
        y_pred = model(x,graphs)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad()
def test(model, test_iter, loss,scaler):
    model.eval()
    l_sum, n = 0.0, 0
    for x,y,graphs in test_iter:
        y_pred = model(x,graphs)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    MSE = l_sum / n

    mae, mse = [], []
    for x, y,graphs in test_iter:
        y = scaler.inverse_transform(y.cpu().numpy().reshape(-1,1)).reshape(-1)
        y_pred = scaler.inverse_transform(model(x,graphs).cpu().numpy().reshape(-1,1)).reshape(-1)
        d = np.abs(y - y_pred)
        mae += d.tolist()
        # mape += (d / y).tolist()
        mse += (d ** 2).tolist()
    MAE = np.array(mae).mean()
    RMSE = np.sqrt(np.array(mse).mean())

    print(f'Dataset PM2.5 | Test loss {MSE:.6f} | MAE {MAE:.6f} | RMSE {RMSE:.6f} ')
    return torch.tensor(l_sum / n)


if __name__ == "__main__":
    # Logging
    logging.basicConfig(filename='DGGNN.log', level=logging.INFO)
    logger = logging.getLogger('DGGNN')

    args, device = get_parameters()
    zscore,d,node_num, train_iter, val_iter, test_iter = data_prepare(args,device)
    x,y,wind = train_iter.dataset[0]
    loss,model,optimizer,es = prepare_model(args,d,node_num,device)
    train(loss, args, optimizer, model, train_iter, val_iter,es)
    test(model,test_iter,loss,zscore)