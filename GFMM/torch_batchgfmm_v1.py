# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:57:55 2018

@author: Thanh Tung Khuat

Batch GFMM classifier (training core) - Faster version by only computing similarity among hyperboxes with the same label
Implemented by Pytorch

    BatchGFMMV1(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range, cardin, clusters)
  
    INPUT
        gamma       Membership function slope (default: 1)
        teta        Maximum hyperbox size (default: 1)
        bthres		Similarity threshold for hyperbox concatenation (default: 0.5)
        simil       Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing        Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isDraw      Progress plot flag (default: 1)
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm      Do normalization of input training samples or not?
        norm_range  New ranging of input data after normalization, for example: [0, 1]
        cardin      Input hyperbox cardinalities
        clusters    Identifiers of objects in each input hyperbox 
        
    ATTRIBUTES
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
      
"""

import sys, os
sys.path.insert(0, os.path.pardir)

import ast
import numpy as np
import torch
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from GFMM.torch_basegfmmclassifier import Torch_BaseGFMMClassifier
from functionhelper.torch_membership_calc import torch_memberG, gpu_memberG
from functionhelper.drawinghelper import drawbox
from functionhelper.torch_hyperboxadjustment import torch_isOverlap, torch_modifiedIsOverlap
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from functionhelper import is_Have_GPU, GPU_Computing_Threshold, UNLABELED_CLASS

class Torch_BatchGFMMV1(Torch_BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        Torch_BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
       
        self.bthres = bthres
        self.simil = simil
        
        if simil == 'mid':
            self.sing = sing
        else:
            self.sing = 'max'
        
    
    def fit(self, X_l, X_u, patClassId):
        """
        X_l          Input data lower bounds (rows = objects, columns = features)
        X_u          Input data upper bounds (rows = objects, columns = features)
        patClassId  Input data class labels (crisp)
        """
        
        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
        
        if isinstance(X_l, torch.Tensor) == False:
            X_l = torch.from_numpy(X_l).float()
            X_u = torch.from_numpy(X_u).float()
            patClassId = torch.from_numpy(patClassId).long()
            
        time_start = time.perf_counter()
        
        isUsingGPU = False
        if is_Have_GPU and X_l.size(0) * X_l.size(1) >= GPU_Computing_Threshold:
            self.V = X_l.cuda()
            self.W = X_u.cuda()
            self.classId = patClassId.cuda()
            isUsingGPU = True
        else:
            self.V = X_l
            self.W = X_u
            self.classId = patClassId
       
        # yX, xX = X_l.size()
        
#        if len(self.cardin) == 0 or len(self.clusters) == 0:
#            self.cardin = np.ones(yX)
#            self.clusters = np.empty(yX, dtype=object)
#            for i in range(yX):
#                self.clusters[i] = np.array([i], dtype = np.int32)
#        
        if self.isDraw:
            mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
            drawing_canvas = self.initializeCanvasGraph("GFMM - AGGLO-SM-Fast version")
                
            # plot initial hyperbox
            Vt, Wt = self.pcatransform()
            color_ = np.empty(len(self.classId), dtype = object)
            for c in range(len(self.classId)):
                color_[c] = mark_col[self.classId[c]]
            drawbox(Vt, Wt, drawing_canvas, color_)
            self.delay()
        
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            # calculate class masks
            yX, xX = self.V.size()
            labList = torch.unique(self.classId[self.classId != UNLABELED_CLASS])
            if isUsingGPU == False:
                clMask = torch.zeros((yX, len(labList)), dtype=torch.uint8)
            else:
                clMask = torch.cuda.ByteTensor(yX, len(labList)).fill_(0)
            
            for i in range(len(labList)):
                clMask[:, i] = (self.classId == labList[i]) | (self.classId == UNLABELED_CLASS)
        
        	# calculate pairwise memberships *ONLY* within each class (faster!)
            if isUsingGPU == False:
                b = torch.zeros((yX, yX))
            else:
                b = torch.cuda.FloatTensor(yX, yX).fill_(0)
            
            if isUsingGPU:
                els = torch.arange(len(labList)).cuda()
            else:
                els = torch.arange(len(labList))
            
            for i in els:
                Vi = self.V[clMask[:, i]] # get bounds of patterns with class label i
                Wi = self.W[clMask[:, i]]
                clSize = torch.sum(clMask[:, i]) # get number of patterns of class i
                clIdxs = torch.nonzero(clMask[:, i])[:, 0] # get position of patterns with class label i in the training set
                
                if self.simil == 'short':
                    for j in range(clSize):
                        if isUsingGPU == False:
                            b[clIdxs[j], clIdxs] = torch_memberG(Wi[j], Vi[j], Vi, Wi, self.gamma, self.oper)
                        else:
                            b[clIdxs[j], clIdxs] = gpu_memberG(Wi[j], Vi[j], Vi, Wi, self.gamma, self.oper)
                elif self.simil == 'long':
                    for j in range(clSize):
                        if isUsingGPU == False:
                            b[clIdxs[j], clIdxs] = torch_memberG(Vi[j], Wi[j], Wi, Vi, self.gamma, self.oper)
                        else:
                            b[clIdxs[j], clIdxs] = gpu_memberG(Vi[j], Wi[j], Wi, Vi, self.gamma, self.oper)
                else:
                    for j in range(clSize):
                        if isUsingGPU == False:
                            b[clIdxs[j], clIdxs] = torch_memberG(Vi[j], Wi[j], Vi, Wi, self.gamma, self.oper)
                        else:
                            b[clIdxs[j], clIdxs] = gpu_memberG(Vi[j], Wi[j], Vi, Wi, self.gamma, self.oper)
                
            if yX == 1:
                maxb = torch.FloatTensor([])
            else:
                maxb = self.torch_splitSimilarityMaxtrix(b, self.sing, False, isUsingGPU)
                if len(maxb) > 0:
                    maxb = maxb[(maxb[:, 2] >= self.bthres), :]
                    
                    if len(maxb) > 0:
                        # sort maxb in the decending order following the last column
                        values, idx_smaxb = torch.sort(maxb[:, 2], descending=True)
                        maxb = torch.cat((maxb[idx_smaxb, 0].reshape(-1, 1), maxb[idx_smaxb, 1].reshape(-1, 1), maxb[idx_smaxb, 2].reshape(-1, 1)), dim=1)
                        #maxb = maxb[idx_smaxb]
            
            while len(maxb) > 0:
                curmaxb = maxb[0, :] # current position handling
                
                # calculate new coordinates of curmaxb(0)-th hyperbox by including curmaxb(1)-th box, scrap the latter and leave the rest intact
                newV = torch.cat((self.V[0:curmaxb[0].long(), :], torch.min(self.V[curmaxb[0].long(), :], self.V[curmaxb[1].long(), :]).reshape(1, -1), self.V[curmaxb[0].long() + 1:curmaxb[1].long(), :], self.V[curmaxb[1].long() + 1:, :]), dim=0)
                newW = torch.cat((self.W[0:curmaxb[0].long(), :], torch.max(self.W[curmaxb[0].long(), :], self.W[curmaxb[1].long(), :]).reshape(1, -1), self.W[curmaxb[0].long() + 1:curmaxb[1].long(), :], self.W[curmaxb[1].long() + 1:, :]), dim=0)
                newClassId = torch.cat((self.classId[0:curmaxb[1].long()], self.classId[curmaxb[1].long() + 1:]))
                if (newClassId[curmaxb[0].long()] == UNLABELED_CLASS):
                    newClassId[curmaxb[0].long()] = newClassId[curmaxb[1].long()]
                #print('Type newV = ', newV.type())
                # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                if ((((newW[curmaxb[0].long()] - newV[curmaxb[0].long()]) <= self.teta).all() == True) and (not torch_modifiedIsOverlap(newV, newW, curmaxb[0].long(), newClassId, isUsingGPU))):
                    isTraining = True
                    self.V = newV
                    self.W = newW
                    self.classId = newClassId
                    
#                    self.cardin[int(curmaxb[0])] = self.cardin[int(curmaxb[0])] + self.cardin[int(curmaxb[1])]
#                    self.cardin = np.append(self.cardin[0:int(curmaxb[1])], self.cardin[int(curmaxb[1]) + 1:])
#                            
#                    self.clusters[int(curmaxb[0])] = np.append(self.clusters[int(curmaxb[0])], self.clusters[int(curmaxb[1])])
#                    self.clusters = np.append(self.clusters[0:int(curmaxb[1])], self.clusters[int(curmaxb[1]) + 1:])
#                    
                    # remove joined pair from the list as well as any pair with lower membership and consisting of any of joined boxes
                    mask = (maxb[:, 0] != curmaxb[0]) & (maxb[:, 1] != curmaxb[0]) & (maxb[:, 0] != curmaxb[1]) & (maxb[:, 1] != curmaxb[1]) & (maxb[:, 2] >= curmaxb[2])
                    maxb = maxb[mask, :]
                    
                    # update indexes to accomodate removed hyperbox
                    # indices of V and W larger than curmaxb(1,2) are decreased 1 by the element whithin the location curmaxb(1,2) was removed 
                    if len(maxb) > 0:
                        maxb[maxb[:, 0] > curmaxb[1], 0] = maxb[maxb[:, 0] > curmaxb[1], 0] - 1
                        maxb[maxb[:, 1] > curmaxb[1], 1] = maxb[maxb[:, 1] > curmaxb[1], 1] - 1
                            
                    if self.isDraw:
                        Vt, Wt = self.pcatransform()
                        color_ = np.empty(len(self.classId), dtype = object)
                        for c in range(len(self.classId)):
                            color_[c] = mark_col[self.classId[c]]
                        drawing_canvas.cla()
                        drawbox(Vt, Wt, drawing_canvas, color_)
                        self.delay()
                else:
                    maxb = maxb[1:, :]  # scrap examined pair from the list
            
            if isTraining == True and isUsingGPU == True and self.V.size(0) * self.V.size(1) < GPU_Computing_Threshold:
                isUsingGPU = False
                self.V = self.V.cpu()
                self.W = self.W.cpu()
                self.classId = self.classId.cpu()
                
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
         
        return self

                  
if __name__ == '__main__':
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + True: drawing hyperboxes during the training process
          + False: no drawing
    arg5: + Maximum size of hyperboxes (teta, default: 1)
    arg6: + gamma value (default: 1)
    arg7: + Similarity threshold (default: 0.5)
    arg8: + Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg9: + operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg10: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg11: + range of input values after normalization (default: [0, 1])   
    arg12: + Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
    """
    # Init default parameters
    if len(sys.argv) < 5:
        isDraw = False
    else:
        isDraw = string_to_boolean(sys.argv[4])
    
    if len(sys.argv) < 6:
        teta = 1    
    else:
        teta = float(sys.argv[5])
    
    if len(sys.argv) < 7:
        gamma = 1
    else:
        gamma = float(sys.argv[6])
    
    if len(sys.argv) < 8:
        bthres = 0.5
    else:
        bthres = float(sys.argv[7])
    
    if len(sys.argv) < 9:
        simil = 'mid'
    else:
        simil = sys.argv[8]
    
    if len(sys.argv) < 10:
        oper = 'min'
    else:
        oper = sys.argv[9]
    
    if len(sys.argv) < 11:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[10])
    
    if len(sys.argv) < 12:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[11])
        
    if len(sys.argv) < 13:
        sing = 'max'
    else:
        sing = sys.argv[12]
        
    if sys.argv[1] == '1':
        training_file = sys.argv[2]
        testing_file = sys.argv[3]

        # Read training file
        Xtr, X_tmp, patClassIdTr, pat_tmp = loadDataset(training_file, 1, False)
        # Read testing file
        X_tmp, Xtest, pat_tmp, patClassIdTest = loadDataset(testing_file, 0, False)
    
    else:
        dataset_file = sys.argv[2]
        percent_Training = float(sys.argv[3])
        Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(dataset_file, percent_Training, False)
    
    classifier = Torch_BatchGFMMV1(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict_torch(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")