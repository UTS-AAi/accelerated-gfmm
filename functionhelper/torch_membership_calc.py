# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:45:13 2018

@author: Thanh Tung Khuat

Fuzzy membership calculation using Pytorch

Note: On GPU, using Float is faster than using Double
"""
import torch
import numpy as np
from functionhelper import float_def, GPU_Computing_Threshold, is_Have_GPU

def torch_memberG(Xl, Xu, V1, W1, g, oper = 'min', isUsingGPU = False):
    """
    Function for membership calculation
    
        b = torch_memberG(X_l, X_u, V, W, g, oper)
 
   INPUT
     Xl        Input data lower bounds (a row vector with columns being features)
     Xu        Input data upper bounds (a row vector with columns being features)
     V1          Hyperbox lower bounds
     W1          Hyperbox upper bounds
     g          User defined sensitivity parameter 
     oper       Membership calculation operation: 'min' or 'prod' (default: 'min')
  
   OUTPUT
     b			Degrees of membership of the input pattern

   DESCRIPTION
    	Function provides the degree of membership b of an input pattern X (in form of upper bound Xu and lower bound Xl)
        in hyperboxes described by min points V and max points W. The sensitivity parameter g regulates how fast the 
        membership values decrease when an input pattern is separeted from hyperbox core.

    """
    # transfer values to GPU
    if isUsingGPU == False and is_Have_GPU == True and (W1.size(0) * W1.size(1) >= GPU_Computing_Threshold or Xl.size(0) >= GPU_Computing_Threshold):
        X_l = Xl.cuda()
        X_u = Xu.cuda()
        W = W1.cuda()
        V = V1.cuda()
        isUsingGPU = True
    else:
        X_l = Xl
        X_u = Xu
        W = W1
        V = V1
    
    if isinstance(g, torch.Tensor) == False:
        if isinstance(g, np.ndarray) == False:
            if isUsingGPU == True:
                g = torch.cuda.FloatTensor([g])
            else:
                g = torch.FloatTensor([g])
        else:
            if isUsingGPU == True:
                g = torch.cuda.FloatTensor(torch.from_numpy(g).float())
            else:
                g = torch.from_numpy(g).float()
    
    yW = W.size(0)
    
    if isUsingGPU == True:
        onesMat = torch.cuda.FloatTensor(yW, 1).fill_(1)
    else:
        onesMat = torch.ones((yW, 1)).float()
    
    violMax = 1 - torch_fofmemb(onesMat * X_u - W, g, isUsingGPU)
    violMin = 1 - torch_fofmemb(V - onesMat * X_l, g, isUsingGPU)
    
    if oper == 'prod':
        b = torch.prod(torch.min(violMax, violMin), dim = 1)
    else:
        b = torch.min(violMax, violMin).min(dim = 1)[0]
    
    return b

def gpu_memberG(X_l, X_u, V, W, g, oper = 'min'):
    """
    Function for membership calculation
    
        b = torch_memberG(X_l, X_u, V, W, g, oper)
 
   INPUT
     X_l        Input data lower bounds (a row vector with columns being features)
     X_u        Input data upper bounds (a row vector with columns being features)
     V          Hyperbox lower bounds
     W          Hyperbox upper bounds
     g          User defined sensitivity parameter 
     oper       Membership calculation operation: 'min' or 'prod' (default: 'min')
  
   OUTPUT
     b			Degrees of membership of the input pattern

   DESCRIPTION
    	Function provides the degree of membership b of an input pattern X (in form of upper bound Xu and lower bound Xl)
        in hyperboxes described by min points V and max points W. The sensitivity parameter g regulates how fast the 
        membership values decrease when an input pattern is separeted from hyperbox core.

    """
    # transfer values to GPU
    
    if isinstance(g, torch.Tensor) == False:
        if isinstance(g, np.ndarray) == False:
            g = torch.cuda.FloatTensor([g])
        else:
            g = torch.cuda.FloatTensor(torch.from_numpy(g).float())
    
    yW = W.size(0)
    
    onesMat = torch.cuda.FloatTensor(yW, 1).fill_(1)
    
    violMax = 1 - torch_fofmemb(onesMat * X_u - W, g, True)
    violMin = 1 - torch_fofmemb(V - onesMat * X_l, g, True)
    
    if oper == 'prod':
        b = torch.prod(torch.min(violMax, violMin), dim = 1)
    else:
        b = torch.min(violMax, violMin).min(dim = 1)[0]
    
    return b


def torch_fofmemb(x, gama, isUsingGPU):

    """
    fofmemb - ramp threshold function for fuzzy membership calculation

        f = fofmemb(x,gama)
  
   INPUT
     x			Input data matrix (rows = objects, columns = attributes)
     gama		Steepness of membership function
     isUsingGPU Checking whether using GPU or not
  
   OUTPUT
     f			Fuzzy membership values

   DESCRIPTION
    	f = 1,     if x*gama > 1
    	x*gama,    if 0 =< x*gama <= 1
    	0,         if x*gama < 0
    """

    if gama.size(0) > 1:
        if isUsingGPU == True:
            onesMat = torch.cuda.FloatTensor(x.size(0), 1).fill_(1)
        else:
            onesMat = torch.ones((x.size(0), 1))
            
        p = x * (onesMat * gama)
    else:
        p = x * gama[0]

    if isUsingGPU:
        f = ((p >= 0) * (p <= 1)).type(torch.cuda.FloatTensor) * p + (p > 1).type(torch.cuda.FloatTensor)
    else:
        f = ((p >= 0) * (p <= 1)).type(torch.float)* p + (p > 1).type(torch.float)
    
    return f

def torch_asym_similarity_one_many(Xl_k, Xu_k, V, W, g = 1, asym_oper = 'max', oper_mem = 'min', isUsingGPU = False):
    """
    Calculate the asymetrical similarity value of the k-th hyperbox (lower bound - Xl_k, upper bound - Xu_k) and 
    hyperboxes having lower and upper bounds stored in two matrix V and W respectively
    
    INPUT
        Xl_k        Lower bound of the k-th hyperbox
        Xu_k        Upper bound of the k-th hyperbox
        V           Lower bounds of other hyperboxes
        W           Upper bounds of other hyperboxes
        g           User defined sensitivity parameter 
        asym_oper   Use 'min' or 'max' (default) to compute the asymetrical similarity value
        oper_mem    operator used to compute the membership value, 'min' or 'prod'
        
    OUTPUT
        b           similarity values of hyperbox k with all hyperboxes having lower and upper bounds in V and W
    
    """
    isUsingGPULocal = isUsingGPU
    if isUsingGPU == False and W.size(0) * W.size(1) >= GPU_Computing_Threshold:
        isUsingGPULocal = True
        V = V.cuda()
        W = W.cuda()
        Xl_k = Xl_k.cuda()
        Xu_k = Xu_k.cuda()
        
    numHyperboxes = W.size(0)
    
    if isinstance(g, torch.Tensor) == False:
        if isinstance(g, np.ndarray) == False:
            if isUsingGPULocal == True:
                g = torch.cuda.FloatTensor([g])
            else:
                g = torch.FloatTensor([g])
        else:
            if isUsingGPULocal == True:
                g = torch.cuda.FloatTensor(torch.from_numpy(g).float())
            else:
                g = torch.from_numpy(g).float()
    
    Vk = Xl_k.expand(numHyperboxes, Xl_k.size(0))
    Wk = Xu_k.expand(numHyperboxes, Xu_k.size(0))
    
    violMax1 = 1 - torch_fofmemb(Wk - W, g, isUsingGPULocal)
    violMin1 = 1 - torch_fofmemb(V - Vk, g, isUsingGPULocal)
    
    violMax2 = 1 - torch_fofmemb(W - Wk, g, isUsingGPULocal)
    violMin2 = 1 - torch_fofmemb(Vk - V, g, isUsingGPULocal)
    
    if oper_mem == 'prod':
        b1 = torch.prod(torch.min(violMax1, violMin1), dim = 1)
        b2 = torch.prod(torch.min(violMax2, violMin2), dim = 1)
    else:
        b1 = torch.min(violMax1, violMin1).min(dim = 1)[0]
        b2 = torch.min(violMax2, violMin2).min(dim = 1)[0]
        
    if asym_oper == 'max':
        b = torch.max(b1, b2)
    else:
        b = torch.min(b1, b2)
    
    if isUsingGPU == False and isUsingGPULocal == True:
        b = b.cpu()
    
    return b