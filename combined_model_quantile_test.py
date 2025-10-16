# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 00:41:49 2021

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
import math

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
    df = pd.read_csv(args.data_path+'college_data_sup.csv', encoding="utf-8",sep=',', header=None)
    min_ = [1,2,3,4,5,6,7,8,9]
    learning_rate = 0.005
    num_epochs = 10000
    upper = 100
    lower = 40
    lamb = 0.5
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
            float_arr[:,i] = scale_to_01_range(1/(1e-2 + float_arr[:,i]))
            float_arr[b,i] = 0
            continue
        float_arr[:,i] = scale_to_01_range(1/(1e-2 + float_arr[:,i]))
        

X_total=float_arr[:,:-1]
y = float_arr[:,-1]
#X=X_total[:,1:]
X_train_total, X_test_total, y_train, y_test = train_test_split(X_total, y, test_size=0.3, random_state=42)
X = X_test_total[:,1:]

params = {'batch_size': 64,
          'shuffle': True}

partition = dict()
partition['train'] =X_train_total[:,0]
partition['validation']= X_test_total[:,0]
data = dict()
data['train'] =X_train_total[:,1:]
data['validation']= X_test_total[:,1:]

X = Variable(torch.from_numpy(X.astype(np.float32)))
x_test = Variable(torch.from_numpy(X_test_total[:,1:].astype(np.float32)))
training_set = Dataset(partition['train'], data['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params)

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
        self.relu = nn.ELU()
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
        #x = nn.functional.linear(x,self.log_weight1.exp())
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
c_loss = 100
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
        temp_loss4 = 0
        temp_loss5 = 0
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
                if(i==7):
                    continue
                temp_loss1 += mean_grad[i]/mean_grad[7]
            for i in range(features):
                if(i==7 or i==4):
                    continue
                temp_loss2+= mean_grad[i]/mean_grad[4]
            for i in range(features):
                if(i==7 or i==4 or i==5 or i==6):
                    continue
                temp_loss3+= mean_grad[i]/(mean_grad[5]*mean_grad[6])
            for i in range(features):
                if(i==7 or i==4 or i==5 or i==6 or i==1 or i==2 or i==3 or i==8):
                    continue
                temp_loss4 += mean_grad[i]/(mean_grad[2])
            for i in range(features):
                temp_loss5 += mean_grad[i]/mean_grad[0]
     
        temp_loss7 = torch.mean(torch.max(torch.zeros(batchsize,1),so-upper*torch.ones(batchsize,1)),0)
        temp_loss6 = torch.mean(torch.max(torch.zeros(batchsize,1),lower*torch.ones(batchsize,1)-so),0) 
        
        dev1 = torch.std(so)
        mean1 = torch.mean(so)
        
        
        quant_loss1 = torch.max(torch.zeros(1,1),torch.abs(0.67*dev1+mean1-50))
        quant_loss2 = torch.max(torch.zeros(1,1),torch.abs(-2.32*dev1+mean1-40))
        
        
        if(args.dataset_name=='movie'):
             loss = temp_loss7 + temp_loss6 + 0.1 * temp_loss1
        elif(args.dataset_name=='college'):
            loss =  quant_loss1 + quant_loss2 + temp_loss7 + 2*temp_loss6 + temp_loss1 + temp_loss2+temp_loss3 + temp_loss4 - 0.1*temp_loss5
            #loss = kl_loss
        
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
            print(temp_loss4)
            print(temp_loss6)
            print(temp_loss7)
            print(quant_loss1)
            print(quant_loss2)
            print(loss)
            #backward propagation: calculate gradients
            predicted = model(X).detach().numpy()
#plt.plot(np.arange(0,epoch+1),loss_list)
#plt.savefig(args.data_path+args.dataset_name+'/fig{}.png'.format(args.exp_name))
            print("Test MSE of unsupervised is : {}".format(np.sqrt(np.mean((predicted-y_test)**2))))
            print('***************')
            if(loss<c_loss):
                c_loss = loss
                predicted_ = predicted

csv_file = np.hstack((X_test_total,predicted_.reshape(-1,1)))
csv_file = np.hstack((csv_file,y_test.reshape(-1,1)))
output_dir = args.data_path+args.dataset_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_dir+'/quant_{}.csv'.format(args.exp_name), 'w', encoding="utf-8") as f:
    csv.writer(f).writerows(csv_file)
