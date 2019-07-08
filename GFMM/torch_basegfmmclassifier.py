# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 22:52:51 2018

@author: Thanh Tung Khuat

Base GFMM classifier. Implemented by Pytorch
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GFMM.classification import torch_predict
from functionhelper.matrixhelper import delete_const_dims, pca_transform
from functionhelper.preprocessinghelper import normalize
from functionhelper import device, float_def, long_def, GPU_Computing_Threshold, is_Have_GPU

class Torch_BaseGFMMClassifier(object):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.gamma = gamma
        self.teta = teta
        self.isDraw = isDraw
        self.oper = oper
        self.isNorm = isNorm
        
        self.V = torch.FloatTensor([])
        self.W = torch.FloatTensor([])
        self.classId = torch.FloatTensor([])
      
        # parameters for data normalization
        self.loLim = norm_range[0]
        self.hiLim = norm_range[1]
        self.mins = torch.FloatTensor([])
        self.maxs = torch.FloatTensor([])
        self.delayConstant = 0.001 # delay time period to display hyperboxes on the canvas
    
    def dataPreprocessing(self, X_l, X_u):
        """
        Preprocess data: delete constant dimensions, Normalize input samples if needed
        
        INPUT:
            X_l          Input data lower bounds (rows = objects, columns = features) datatype: numpy array
            X_u          Input data upper bounds (rows = objects, columns = features) datatype: numpy array
        
        OUTPUT
            X_l, X_u were preprocessed
        """
        
        # delete constant dimensions
        #X_l, X_u = delete_const_dims(X_l, X_u)
        
        # Normalize input samples if needed
        if X_l.min() < self.loLim or X_u.min() < self.loLim or X_u.max() > self.hiLim or X_l.max() > self.hiLim:
            self.mins = torch.from_numpy(X_l.min(axis = 0)).float() # get min value of each feature
            self.maxs = torch.from_numpy(X_u.max(axis = 0)).float() # get max value of each feature
            X_l = normalize(X_l, [self.loLim, self.hiLim])
            X_u = normalize(X_u, [self.loLim, self.hiLim])
        else:
            self.isNorm = False
            self.mins = torch.FloatTensor([])
            self.maxs = torch.FloatTensor([])
            
        return (X_l, X_u)
    
    
    def torch_pcatransform(self):
        """
        Perform PCA transform of V and W if the dimensions are larger than 3
        
        OUTPUT:
            V and W in the new space
        """
        yX, xX = self.V.size()
                
        if (xX > 3):
            Vt = pca_transform(self.V.numpy(), 3)
            Wt = pca_transform(self.W.numpy(), 3)
            mins = Vt.min(axis = 0)
            maxs = Wt.max(axis = 0)
            Vt = self.loLim + (self.hiLim - self.loLim) * (Vt - np.ones((yX, 1)) * mins.numpy()) / (np.ones((yX, 1)) * (maxs - mins))
            Wt = self.loLim + (self.hiLim - self.loLim) * (Wt - np.ones((yX, 1)) * mins.numpy()) / (np.ones((yX, 1)) * (maxs - mins))
        else:
            Vt = self.V.numpy()
            Wt = self.W.numpy()
            
        return (Vt, Wt)
    
    def initializeCanvasGraph(self, figureName, numDims):
        """
        Initialize canvas to draw hyperbox
        
            INPUT
                figureName          Title name of windows containing hyperboxes
                numDims             The number of dimensions of hyperboxes
                
            OUTPUT
                drawing_canvas      Plotting object of python
        """
        fig = plt.figure(figureName)
        plt.ion()
        if numDims == 2:
            drawing_canvas = fig.add_subplot(1, 1, 1)
            drawing_canvas.axis([0, 1, 0, 1])
        else:
            drawing_canvas = Axes3D(fig)
            drawing_canvas.set_xlim3d(0, 1)
            drawing_canvas.set_ylim3d(0, 1)
            drawing_canvas.set_zlim3d(0, 1)
            
        return drawing_canvas
    
    def delay(self):
        """
        Delay a time period to display hyperboxes
        """
        plt.pause(self.delayConstant)
    
    
    def torch_rot90(self, A):
        res = torch.transpose(torch.flip(A, [1]), 0, 1)
        return res
    
        
    def torch_splitSimilarityMaxtrix(self, A, asimil_type = 'max', isSort = True, isUsingGPU = False):
        """
        Split the similarity matrix A into the maxtrix with three columns:
            + First column is row indices of A
            + Second column is column indices of A
            + Third column is the values corresponding to the row and column
        
        if isSort = True, the third column is sorted in the descending order 
        
            INPUT
                A               Degrees of membership of input patterns (each row is the output from memberG function)
                asimil_type     Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
                isSort          Sorting flag
                
            OUTPUT
                The output as mentioned above
        """
        isUsingGPULocal = isUsingGPU
        if isUsingGPU == False and is_Have_GPU == True and A.size(0) * A.size(1) >= GPU_Computing_Threshold:
            A = A.cuda()
            isUsingGPULocal = True
            
        # get min/max memberships from triu and tril of memberhsip matrix which might not be symmetric (simil=='mid')
        if asimil_type == 'min':
            transformedA = torch.min(torch.flip(self.torch_rot90(torch.tril(A, diagonal=-1)), [0]), torch.triu(A, diagonal=1))  # rotate tril to align it with triu for min (max) operation
        else:
            transformedA = torch.max(torch.flip(self.torch_rot90(torch.tril(A, diagonal=-1)), [0]), torch.triu(A, diagonal=1))
        
        ind_rows_columns = torch.nonzero(transformedA)
        values = transformedA[transformedA != 0]
        
        if isSort == True:
            sortedTransformedA, ind_SortedTransformedA = torch.sort(values, descending=True)
            if isUsingGPULocal == True:
                result = torch.cat((ind_rows_columns[ind_SortedTransformedA].type(torch.cuda.FloatTensor), sortedTransformedA.reshape(-1, 1)), dim=1)
            else:
                result = torch.cat((ind_rows_columns[ind_SortedTransformedA].type(torch.float), sortedTransformedA.reshape(-1, 1)), dim=1)
        else:
            if isUsingGPULocal == True:
                result = torch.cat((ind_rows_columns.type(torch.cuda.FloatTensor), values.reshape(-1, 1)), dim=1)
            else:
                result = torch.cat((ind_rows_columns.type(torch.float), values.reshape(-1, 1)), dim=1)
        
        if isUsingGPU == False and isUsingGPULocal == True:
            result = result.cpu()
            
        return result
    
    
    def predict_torch(self, Xl_Test, Xu_Test, patClassIdTest):
        """
        Perform classification
        
            result = predict(Xl_Test, Xu_Test, patClassIdTest)
        
        INPUT:
            Xl_Test             Test data lower bounds (rows = objects, columns = features)
            Xu_Test             Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	     Test data class labels (crisp)
            
        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships
        """
        #Xl_Test, Xu_Test = delete_const_dims(Xl_Test, Xu_Test)
        # Normalize testing dataset if training datasets were normalized
        if isinstance(Xl_Test, torch.Tensor) == False:
            Xl_Test = torch.from_numpy(Xl_Test).float()
            Xu_Test = torch.from_numpy(Xu_Test).float()
            patClassIdTest = torch.LongTensor(patClassIdTest)
            
        if len(self.mins) > 0:
            noSamples = Xl_Test.size(0)
            Xl_Test = self.loLim + (self.hiLim - self.loLim) * (Xl_Test - torch.ones((noSamples, 1)) * self.mins) / (torch.ones((noSamples, 1)) * (self.maxs - self.mins))
            Xu_Test = self.loLim + (self.hiLim - self.loLim) * (Xu_Test - torch.ones((noSamples, 1)) * self.mins) / (torch.ones((noSamples, 1)) * (self.maxs - self.mins))
            
            if Xl_Test.min() < self.loLim or Xu_Test.min() < self.loLim or Xl_Test.max() > self.hiLim or Xu_Test.max() > self.hiLim:
                print('Test sample falls outside', self.loLim, '-', self.hiLim, 'interval')
                print('Number of original samples = ', noSamples)
                
                # only keep samples within the interval loLim-hiLim
                indXl_good = torch.nonzero((Xl_Test >= self.loLim).all(dim = 1) & (Xl_Test <= self.hiLim).all(dim = 1))
                indXu_good = torch.nonzero((Xu_Test >= self.loLim).all(dim = 1) & (Xu_Test <= self.hiLim).all(dim = 1))
                indKeep = torch.from_numpy(np.intersect1d(indXl_good.numpy(), indXu_good.numpy())).long()
                
                Xl_Test = Xl_Test[indKeep, :]
                Xu_Test = Xu_Test[indKeep, :]
                
                print('Number of kept samples =', Xl_Test.shape[0])
                #return
            
        # do classification
        result = None
        
        if Xl_Test.size(0) > 0:
            result = torch_predict(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)
            
        return result
