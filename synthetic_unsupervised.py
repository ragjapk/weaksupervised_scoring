# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:44:45 2021

@author: 1832157
"""

#score between 0 and 9
#dropout present
#losses for lead gen rate, sum(leads, CTR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
import os
import csv
from statistics import mean
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scale_to_11_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return 2*(starts_from_zero / value_range)-1


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, data):
        'Initialization'
        self.data = data
        self.list_IDs = list_IDs
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.list_IDs[index]
    
        # Load data and get label
        X = Variable(torch.from_numpy(np.array(self.data[index]).astype(np.float32)),requires_grad=True)
        return X
    
class SuperDataset(torch.utils.data.Dataset):
    def __init__(self, xtrain, ytrain):
        'Initialization'
        self.xtrain = xtrain
        self.ytrain = ytrain
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.xtrain)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = Variable(torch.from_numpy(np.array(self.xtrain[index]).astype(np.float32)),requires_grad=True)
        y = Variable(torch.from_numpy(np.array(self.ytrain[index]).astype(np.float32)))
        return X, y

parser = argparse.ArgumentParser(description='Script to run ad effectiveness model')
parser.add_argument('--exp_name', type=str, default='none')
parser.add_argument('--data_path', help='path where results are stored', default='')
parser.add_argument('--dataset_name', type=str, default='movie')
#parser.add_argument('--data_file', help='name of the data file', default='data_euclidean.csv')
args = parser.parse_args()
    
if args.dataset_name =='synth':
    df = pd.read_csv(args.data_path+'synthetic.csv', encoding="utf-8",sep=',', header=None)   
    min_ =[]
    num_epochs =2000
    upper = 700
    lower = 15
    dev2 = 78
    mean2 = 320
    lamb = 0.5
    learning_rate = 0.001
    float_arr=df.values.copy()   

elif args.dataset_name =='synth2':
    df = pd.read_csv(args.data_path+'synthetic2.csv', encoding="utf-8",sep=',', header=None)   
    min_ =[]
    num_epochs =5000
    upper = 26
    lower = 0
    dev2 = 5
    mean2 = 13
    lamb = 0.5
    learning_rate = 0.001
    float_arr=df.values.copy()

elif args.dataset_name =='synth3':
    df = pd.read_csv(args.data_path+'synthetic3.csv', encoding="utf-8",sep=',', header=None)   
    min_ =[]
    num_epochs =5000
    upper = 102
    lower = 80
    dev2 = 4
    mean2 = 91
    lamb = 0.5
    learning_rate = 0.001
    float_arr=df.values.copy() 

elif args.dataset_name =='synth4':
    df = pd.read_csv(args.data_path+'synthetic4.csv', encoding="utf-8",sep=',', header=None)   
    min_ =[]
    num_epochs =5000
    upper = 100
    lower = 0
    dev2 = 11
    mean2 = 16
    lamb = 2
    learning_rate = 0.001
    float_arr=df.values.copy()
    
elif args.dataset_name =='synth5':
    df = pd.read_csv(args.data_path+'synthetic_normalized.csv', encoding="utf-8",sep=',', header=None)   
    min_ =[]
    num_epochs =5000
    upper = 60
    lower = 0
    dev2 = 10
    mean2 = 15
    lamb = 2
    learning_rate = 0.001
    float_arr=df.values.copy()
#Performing 1-x for negative features
if(min_):
    for i in min_:
        if(args.dataset_name=='ads' and (i==14 or i==5 or i==1 or i==6 or i==17)):
            b = np.where(float_arr[:,i] == 0)
            #float_arr[:,i] = scale_to_01_range(1/(1e-2 + float_arr[:,i]))
            float_arr[:,i] = 1-float_arr[:,i]
            float_arr[b,i] = 0
            continue
        #float_arr[:,i] = scale_to_01_range(1/(1e-2 + float_arr[:,i]))
        
        if(args.dataset_name=='_movie' or args.dataset_name=='_college' or args.dataset_name=='_journal'):
            float_arr[:,i] = scale_to_11_range(1-float_arr[:,i])
        else:
            float_arr[:,i] = 1-float_arr[:,i]

#X_total=float_arr[:,:]
X_total=float_arr[:,:-1]
y = float_arr[:,-1]


X_train_total, X_test_total, y_train, y_test = train_test_split(X_total, y, test_size=0.3, random_state=42)

#X_train_total, X_test = X_train_total.to(device), X_test_total.to(device)
#y_train, y_test = y_train.to(device), y_test.to(device)


data2 = dict()
data2['xtrain'] =X_train_total[:,:]
data2['ytrain'] = y_train[:]
X = X_test_total

partition = dict()
partition['train'] =X_train_total[:,0]
partition['validation']= X_test_total[:,0]

data = dict()
data['train'] =X_train_total[:,:]
data['validation']= X_test_total[:,:]


params = {'batch_size': 64,
          'shuffle': True}

params2 = {'batch_size': 64,
          'shuffle': True}


training_set2 = SuperDataset(data2['xtrain'], data2['ytrain'])
training_generator2 = torch.utils.data.DataLoader(training_set2, **params2)

X = Variable(torch.from_numpy(X.astype(np.float32)))
X = X.to(device)
training_set = Dataset(partition['train'], data['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params)

class Network(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.log_weight1 = nn.Parameter(torch.Tensor(500, input_size))
        #self.log_weight2 = nn.Parameter(torch.Tensor(128,500))
        #self.log_weight3 = nn.Parameter(torch.Tensor(500,128))
        self.log_weight4 = nn.Parameter(torch.Tensor(256,500))
        self.log_weight5 = nn.Parameter(torch.Tensor(1,256))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight1)
        #nn.init.xavier_uniform_(self.log_weight2)
        #nn.init.xavier_uniform_(self.log_weight3)
        nn.init.xavier_uniform_(self.log_weight4)
        nn.init.xavier_uniform_(self.log_weight5)
    
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = nn.functional.linear(x,self.log_weight1)
        x = self.relu(x)
        #x = nn.functional.linear(x,self.log_weight2)
        #x = self.relu(x)
        #x = nn.functional.linear(x,self.log_weight3)
        #x = self.relu(x)
        x = nn.functional.linear(x,self.log_weight4)
        pre_x = self.relu(x)
        x = nn.functional.linear(pre_x,self.log_weight5)
        return pre_x,x

    
for batch in training_generator:
    length = batch.shape[1]
    batchsize = batch.shape[0]
    break

#for sup_batch, sup_label in training_generator2:
    #print(sup_label)

model = Network(length)
model.to(device)
l1 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )
c_loss = 500000
for epoch in range(num_epochs):
    for local_data in training_generator:
        #forward feed
        batchsize = local_data.shape[0]
        features = local_data.shape[1]
        local_data = local_data.to(device)
        act,so = model(local_data)
        gradi = torch.autograd.grad(outputs=so, inputs=local_data, grad_outputs=torch.ones_like(so),retain_graph=True, create_graph=True)[0]
        mean_grad = torch.mean(gradi,0)
        #calculate the losss
        temp_loss1 = 0
        temp_loss2 = 0
        temp_loss3 = 0
        temp_loss4 = 0
        temp_loss5 = 0
        
        if(args.dataset_name=='synth'):
            for i in range(features):
                if(i==3):
                    continue
                temp_loss1 += mean_grad[i]/mean_grad[3]  
                if(i==3 or i==2):
                    continue
                temp_loss2 += mean_grad[i]/mean_grad[2]
                if(i==3 or i==2 or i==1):
                    continue
                temp_loss3 += mean_grad[i]/(mean_grad[1])    

        temp = so-(upper*torch.cuda.FloatTensor(batchsize, 1).fill_(1))
        #temp_loss7 = torch.mean(torch.exp(temp),0)
        temp_loss7 = torch.mean(torch.maximum(torch.cuda.FloatTensor(batchsize, 1).fill_(0),temp),0)
        #temp_loss72 = torch.mean(torch.max(torch.zeros(batchsize,1),torch.exp(temp)),0)
        temp_loss6 = torch.mean(torch.maximum(torch.cuda.FloatTensor(batchsize, 1).fill_(0),torch.square(((mean2)*torch.cuda.FloatTensor(batchsize, 1).fill_(1))-so)),0) 
        #temp_loss8 = torch.mean(torch.max(torch.zeros(batchsize,1),torch.abs(upper*torch.ones(batchsize,1)-so)),0)
        temp2 = (lower*torch.cuda.FloatTensor(batchsize, 1).fill_(1))-so
        #temp_loss8 = torch.mean(torch.exp(temp2),0)
        temp_loss8 = torch.mean(torch.maximum(torch.cuda.FloatTensor(batchsize, 1).fill_(0),temp2),0)
        #temp_loss92 = torch.mean(torch.max(torch.zeros(batchsize,1),torch.exp(temp)),0)
        dev1 = torch.std(so)*(batchsize/(batchsize-1))
        mean1 = torch.mean(so)
        
        dev2 = torch.cuda.FloatTensor(1, 1).fill_(78)
        mean2 = torch.cuda.FloatTensor(1, 1).fill_(320)
        
        kl_loss = torch.log(dev1/dev2) + ((dev2*dev2) + (mean1-mean2)**2)/(2*dev1*dev1) - 0.5
        
        #kl_loss = -torch.log(torch.tensor(lamb).cuda()) + lamb*mean1 - 0.5 - 0.5*torch.log(2*math.pi*dev1*dev1) - lamb*40

        #quant_loss1 = torch.max(torch.cuda.FloatTensor(1, 1).fill_(0),torch.abs(0.67*dev1+mean1-50).cuda())
        
        loss = kl_loss + temp_loss1 + temp_loss2 + temp_loss3 + temp_loss7 + temp_loss8
            #temp_loss1 + temp_loss2 + temp_loss7 + temp_loss8 + temp_loss6 + quant_loss1
        #if(epoch>8000):     
        loss.backward()
    
        #update the weights
        optimizer.step()
    
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        
        #with torch.no_grad():
            #for name, param in model.state_dict().items():
                #if ("weight" in name):
                    #transformed_param = torch.where(param>0,param,torch.pow(2.718, param-5))
                    #model.state_dict()[name].copy_(transformed_param)
    if(epoch%100==0):
        print('################')
        print(epoch)
        print(mean1,mean2)
        print(dev1,dev2)
        print(loss)
        #backward propagation: calculate gradients
    if(loss<c_loss):
        c_loss = loss
        act1,pred = model(X)
        predicted_ = pred.detach().cpu().numpy()
print("Test MSE of unsupervised is : {}".format(np.sqrt(np.mean((predicted_-y_test)**2))))
print("Best loss is : {}".format(c_loss))
print('***************')
csv_file = np.hstack((X_test_total,predicted_.reshape(-1,1)))
csv_file = np.hstack((csv_file,y_test.reshape(-1,1)))
output_dir = args.data_path+args.dataset_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir+'/exp_{}.csv'.format(args.exp_name), 'w', encoding="utf-8") as f:
    csv.writer(f).writerows(csv_file)
