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
from torchsummary import summary
import pytorch_stats_loss
import math

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
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

parser = argparse.ArgumentParser(description='Script to run ad effectiveness model')
parser.add_argument('--exp_name', type=str, default='none')
parser.add_argument('--data_path', help='path where results are stored', default='')
parser.add_argument('--dataset_name', type=str, default='movie')
#parser.add_argument('--data_file', help='name of the data file', default='data_euclidean.csv')
args = parser.parse_args()

if args.dataset_name == 'movie':
    df = pd.read_csv(args.data_path+'data_euclidean.csv', encoding="utf-8",sep=',', header=None)
    df = df.drop(df.columns[-1],axis=1)
    min_ = [3]
    learning_rate = 0.0005
    num_epochs = 2000
    upper = 10
    lower = 0
    dev2 = 0.5
    mean2 = 5
    lamb = 1
elif args.dataset_name == 'college':
    df = pd.read_csv(args.data_path+'college_data.csv', encoding="utf-8",sep=',', header=None)
    df = df.drop(df.columns[-1],axis=1)
    min_ = [1,2,3,4,5,6,7,8,9]
    learning_rate = 0.0002
    num_epochs = 6000
    upper = 100
    lower = 40
    dev2 = 9
    mean2 = 70
    lamb = 10
elif args.dataset_name == 'ads':
    df = pd.read_csv(args.data_path+'data_euclidean_latest.csv', encoding="utf-8",sep=',', header=None)
    df = df.drop(df.columns[-1],axis=1)
    min_ = [1,5,6,14,17]
    learning_rate = 0.001
    num_epochs = 10000
    upper = 10
    lower=0
    dev2 = 0.5
    mean2 = 5
    lamb = 1


float_arr=df.values.copy()

#Performing 1-x for negative features
if(min_):
    for i in min_:
        if(args.dataset_name=='ads' and (i==14 or i==5 or i==1 or i==6 or i==17)):
            b = np.where(float_arr[:,i] == 0)
            float_arr[:,i] = 1 - float_arr[:,i]
            float_arr[b,i] = 0
            continue
        float_arr[:,i] = 1- float_arr[:,i]
        

X_total=float_arr[:,:]
X=float_arr[:,1:]
X_train_total, X_test_total = train_test_split(X_total, test_size=0.3, random_state=42)

params = {'batch_size': 64,
          'shuffle': True}

partition = dict()
partition['train'] =X_train_total[:,0]
partition['validation']= X_test_total[:,0]
data = dict()
data['train'] =X_train_total[:,1:]
data['validation']= X_test_total[:,1:]

#X_train=X_train_total[:,1:19]
#X_test=X_test_total[:,1:19]
#X_train = Variable(torch.from_numpy(X_train.astype(np.float32)),requires_grad=True)
#X = Variable(torch.from_numpy(X.astype(np.float32)))
#X_test = Variable(torch.from_numpy(X_test.astype(np.float32)))
X = Variable(torch.from_numpy(X.astype(np.float32)))
x_test = Variable(torch.from_numpy(X_test_total[:,1:].astype(np.float32)))
training_set = Dataset(partition['train'], data['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params)

#print(len(data['validation']))
#validation_set = Dataset(partition['validation'], data['validation'],)
#validation_generator = torch.utils.data.DataLoader(validation_set, **params)

class Network(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(input_size, 256)
        self.hidden2 = nn.Linear(256, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 1)
        #Dropout layers in between the linear layers
        #self.drop = nn.Dropout(p=0.3)
        # Define sigmoid activation and softmax output 
        self.relu = nn.ReLU()
        self.log_weight1 = nn.Parameter(torch.Tensor(256, input_size))
        self.log_weight2 = nn.Parameter(torch.Tensor(256,256))
        self.log_weight3 = nn.Parameter(torch.Tensor(1,256))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight1)
        nn.init.xavier_uniform_(self.log_weight2)
        nn.init.xavier_uniform_(self.log_weight3)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = nn.functional.linear(x,self.log_weight1.exp())
        #x = self.drop(x)
        x = self.relu(x)
        x = nn.functional.linear(x,self.log_weight2.exp())
        #x = self.drop(x)
        x = self.relu(x)
        x = nn.functional.linear(x,self.log_weight3.exp())  
        return x

    
for batch in training_generator:
    length = batch.shape[1]
    batchsize = batch.shape[0]
    break
    
model = Network(length)
l1 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )

for epoch in range(num_epochs):
    for local_data in training_generator:
        #forward feed
        batchsize = local_data.shape[0]
        features = local_data.shape[1]
        so = model(local_data)
        gradi = torch.autograd.grad(outputs=so, inputs=local_data, grad_outputs=torch.ones_like(so),retain_graph=True, create_graph=True)[0]
        #gradi = Variable(gradi.data, requires_grad=True)
        mean_grad = torch.mean(gradi,0)
        #calculate the loss
        temp_loss1 = 0
        temp_loss2 = 0
        temp_loss3 = 0
        if(args.dataset_name=='ads'):
            for i in range(features):
                if(i==12):
                    continue
                temp_loss1 += mean_grad[i]/mean_grad[12]
            for i in range(features):
                if(i==3 or i==12 or i==1 or i==10):
                    continue
                temp_loss2 += mean_grad[i]
            temp_loss2 = temp_loss2/ (mean_grad[10] + mean_grad[1] * mean_grad[3])
            for i in range(features):
                if(i==13 or i==12 or i==3 or i==1 or i==10):
                    continue
                temp_loss3 += mean_grad[i]/mean_grad[13]
        elif(args.dataset_name=='movie'):
            for i in range(features):
                if(i==3):
                    continue
                temp_loss1 += mean_grad[i]/mean_grad[3]
        elif(args.dataset_name=='college'):
            for i in range(features):
                if(i==4 or i==5 or i==6):
                    continue
                temp_loss1 += mean_grad[i]/(mean_grad[6]*mean_grad[5]*mean_grad[4])
     
        temp_loss5 = torch.mean(torch.max(torch.zeros(batchsize,1),torch.abs(upper*torch.ones(batchsize,1)-so)),0)
        temp_loss6 = torch.mean(torch.max(torch.zeros(batchsize,1),torch.abs(so-lower*torch.ones(batchsize,1))),0) 
       
        #temp_loss7 = -torch.var(so)
        
        dev1 = torch.std(so)
        mean1 = torch.mean(so)
        
        kl_loss = -torch.log(torch.tensor(lamb)) + lamb*mean1 - 0.5 - 0.5*torch.log(2*math.pi*dev1*dev1)
        
        #kl_loss = torch.log(dev2/dev1) + ((dev1*dev1 + (mean1-mean2)**2)/2*dev2*dev2) - 0.5
        
        #kl_loss = torch.log(dev1/dev2) + ((dev2*dev2 + (mean2-mean1)**2)/2*dev1*dev1) - 0.5
        
        
        #temp_loss8 = pytorch_stats_loss.torch_wasserstein_loss(gauss,so)
        #if(epoch>500):
        #loss =  temp_loss1 + temp_loss2 +  temp_loss3 +  temp_loss5 + kl_loss
        #else:
        #loss =  temp_loss1 + temp_loss2 +  temp_loss3  +  temp_loss5
        if(args.dataset_name=='movie'):
             loss = torch.max(torch.zeros(1,1),kl_loss) + temp_loss5 + temp_loss6 + temp_loss6/temp_loss5 + temp_loss5/temp_loss6 +temp_loss1
        elif(args.dataset_name=='college'):
            loss =  torch.max(torch.zeros(1,1),kl_loss) + temp_loss5 + temp_loss6 + temp_loss6/temp_loss5 + temp_loss5/temp_loss6 + temp_loss1
        
        loss.backward()
    
        #update the weights
        optimizer.step()
    
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        #for p in model.parameters():
            #p.data.clamp_(0)
    if(epoch%100==0):
            print('################')
            print(temp_loss1)
            print(temp_loss2)
            print(temp_loss3)
            print(loss)
            print(temp_loss5)
            print(temp_loss6)
            print(kl_loss)
            #backward propagation: calculate gradients
            print('***************')
#for param in model.parameters():
  #print(param.data)
predicted = model(X).detach().numpy()


csv_file = np.hstack((X_total,predicted.reshape(-1,1)))
output_dir = args.data_path+args.dataset_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(args.data_path+args.dataset_name+'/exp{}.csv'.format(args.exp_name), 'w', encoding="utf-8") as f:
    csv.writer(f).writerows(csv_file)
