# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:59:38 2020

@author: Janik
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import gamma
import warnings


def ispdf(y):
#checks if vector <y> is (approximately) probability distribution.   
    return np.abs(np.sum(y)-1)<10**(-8) and np.all(y>=0)

def checkpdf(function,y):
#checks if vector <y> is (approximately) probability distribution and prints the reason why not.
    msg = ''
    
    if not ispdf(y):
        if np.any(y<0):
            msg += 'Values at indices ' + str(np.where(y<0)) + ' <0. '
        s = np.sum(y)
        if np.abs(s-1)>=10**(-8):
            msg += 'Sum of values: ' + str(s)
        raise BaseException('In '+function.__name__+': y is not a probability density function. ' + msg)
        

def discretize(x,pdf,*args,**kwargs):
#discretizes a continuous probability function <pdf> with parameters <*args,**kwargs> by intergration over intervals <x>
     x = np.append(x,[-np.inf,np.inf])     
     x = np.sort(x)
     
     y = np.empty(len(x)-1)
     
     f = lambda z: pdf(z,*args,**kwargs)
                 
     for i in range(len(x)-1):
         buffer = quad(f,x[i],x[i+1])
         y[i] = buffer[0]
         
     d = np.sum(y)-1
     
     if d != 0:
         y = y - (d/len(y))
     
     return y    
 

def dnorm(x,loc=0,scale=1):
#returns the discretizes standard Gaussian distribution
    return discretize(x,norm.pdf,loc=loc,scale=scale)


def entropy(y):
#returns the entropy for a discrete probability distribution <y>.
    checkpdf(entropy,y)
    
    return -np.sum(y[y>0]*np.log(y[y>0]))

        
def dkl(p,q,replace0=True):
#returns the KL divergence between two discrete probability distributions <p,q>.
#to ensure that DKL is well-defined, set <replace0> to True, to ensure that there is no 0 in the denominator.
    if np.any(p.shape != q.shape):
        raise BaseException('Differently shaped inputs Kullback-Leibler divergence')

    if replace0:
        q[q==0] += 1e-10
        p[p==0] += 1e-10
    elif np.any((q==0) & (p!=0)):
        raise BaseException('KL divergence only defined if everywhere q=0 implies p=0')

    res = np.sum(p[p>0]*(np.log(p[p>0]) - np.log(q[p>0])))
        
    return res


def draw(pdf):
#Draws a sample from the discrete probability distribution <pdf>
    checkpdf(draw,pdf)
    
    cum = np.cumsum(pdf)
    rv = np.random.rand()
    I = np.where(cum >= rv)
    
    return I[0][0]


def normalize(v,axis=None):
#Normalizes a vector <v> to form a probability distribution. If v is a matrix, it is normalized along the dimension <axis>.
    Z = np.sum(v,axis=axis)
    
    if axis!=None:
        shape = np.asarray(v.shape)
        shape[axis] = 1
        Z = Z.reshape(shape)
        tiles = np.ones(shape.shape,dtype=np.int)
        tiles[axis] = v.shape[axis]
        Z = np.tile(Z,tiles)
        
    if np.all(Z != 0):
        return v / Z
    else:
        raise BaseException('In normalize: vector adds up to 0')
        
        
def int_multivar_beta(N):
#Multivariate beta function
    return np.prod(gamma(N)) / gamma(np.sum(N))

def inv_int_multivar_beta(N):
#inverse multivariate beta function
    return gamma(np.sum(N)) / np.prod(gamma(N))
    