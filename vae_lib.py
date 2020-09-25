# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:58:52 2020

@author: Janik
"""

import torch
from torch import nn, optim
from torch.nn import functional as F

class VAE_full(nn.Module):
#Categorical VAE with full integration over parameters.
#Fully connected encoder and decoder functions with one hidden layer each, ReLU activation function.
    
    def __init__(self,data_size,hidden_size,parameter_size):
        
        super().__init__()
        
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.parameter_size = parameter_size
        
        self.fc1 = nn.Linear(data_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,parameter_size)
        self.fc3 = nn.Linear(parameter_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,data_size)
        
    def encode(self, x):        
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x):        
        x = x.view(-1,self.data_size)
        unnormalized_log_q = self.encode(x)
        sample = torch.eye(self.parameter_size)
        unnormalized_log_p = self.decode(sample)
        
        return unnormalized_log_p, unnormalized_log_q
    
class VAE_MC(nn.Module):
#Categorical VAE with Monte Carlo integration over parameters. Uses the Gumbel-Softmax reparametrization to sample from p.
#Fully connected encoder and decoder functions with one hidden layer each, ReLU activation function.
    
    def __init__(self,data_size,hidden_size,parameter_size):
        
        super().__init__()
        
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.parameter_size = parameter_size
        
        self.fc1 = nn.Linear(data_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,parameter_size)
        self.fc3 = nn.Linear(parameter_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,data_size)
        
    def encode(self, x):        
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)     #this is interpreted as log alpha*q(s|o)
    
    def gumbelize(self, logprobs, tau=1, sample_nr=1):        
        logprobs = logprobs.repeat(sample_nr,1)     #Number of samples for Monte Carlo approximation of the expectation
        sample = F.gumbel_softmax(logprobs, tau=tau, hard=True, dim=1)  #samples as if it was argmax, but differentiates as if it were softmax
        return sample

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)     #this is interpreted as log beta*p(o|s)
    
    def forward(self, x):        
        x = x.view(-1,self.data_size)
        unnormalized_log_q = self.encode(x) #dimensions in q: (observation,state)
        sample = self.gumbelize(unnormalized_log_q, sample_nr=4)
        unnormalized_log_p = self.decode(sample) #dimensions in p: (sample,observation)
        
        return unnormalized_log_p, unnormalized_log_q
    

def full_loss(original,unnormalized_log_p,unnormalized_log_q,message,size,DKL_weight=1):
#Loss function for VAE_full. See section 4.2 for details.
    q = F.softmax(unnormalized_log_q,dim=1)
    log_q = torch.log(q)
    log_p = F.log_softmax(unnormalized_log_p,dim=1)
    log_msg = torch.log(message.view(-1,size) + 1e-10)
    
    rep_original = original.repeat(size,1)    
    rep_log_p = log_p.repeat_interleave(original.size()[0],dim=0)
    
    return torch.sum(q*(DKL_weight*(log_q - log_msg) - rep_log_p[rep_original==1].view(q.size())))


def MC_loss(original,unnormalized_log_p,unnormalized_log_q,message,size,DKL_weight=1):
#Loss function for VAE_MC. See section 4.2 for details.
    q = F.softmax(unnormalized_log_q,dim=1)
    log_q = torch.log(q)
    log_p = F.log_softmax(unnormalized_log_p,dim=1)
    log_msg = torch.log(message.view(-1,size) + 1e-10)
    
    sample_nr = int(unnormalized_log_p.size()[0] / original.size()[0])    
    rep_original = original.repeat(sample_nr, 1)
    rep_log_p = log_p[rep_original==1]
    
    return torch.sum(q*(DKL_weight*(log_q - log_msg))) - torch.sum(rep_log_p)
    