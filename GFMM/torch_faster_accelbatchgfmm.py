# -*- coding: utf-8 -*-
"""
Created on November 06

@author: Thanh Tung Khuat

    Accelerated Batch GFMM classifier (training core)
    
    Implemented by Pytorch
    
    Filter all hyperboxes belonging to the same class before computing similarity measures 
        
        AccelBatchGFMM(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
  
    INPUT:
        gamma           Membership function slope (default: 1)
        teta            Maximum hyperbox size (default: 1)
        bthres          Similarity threshold for hyperbox concatenation (default: 0.5)
        simil           Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing            Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        isDraw          Progress plot flag (default: False)
        oper            Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm          Do normalization of input training samples or not?
        norm_range      New ranging of input data after normalization, for example: [0, 1]
        cardin      Input hyperbox cardinalities
        clusters    Identifiers of objects in each input hyperbox 
        
    ATTRIBUTES:
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        # Comment 2 attributes to accelerate the code because of non-using now
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

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

from functionhelper import UNLABELED_CLASS
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from functionhelper.torch_hyperboxadjustment import torch_modifiedIsOverlap
from GFMM.torch_basegfmmclassifier import Torch_BaseGFMMClassifier
from functionhelper.torch_membership_calc import torch_asym_similarity_one_many, torch_memberG, gpu_memberG
from functionhelper import is_Have_GPU, GPU_Computing_Threshold

class Torch_AccelBatchGFMM(Torch_BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        Torch_BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.bthres = bthres
        self.simil = simil
        self.sing = sing
        
        # Currently, we do not yet use cardin and clusters
#        self.cardin = cardin
#        self.clusters = clusters
    
    
    def fit(self, X_l, X_u, patClassId):  
        """
        Xl          Input data lower bounds (rows = objects, columns = features)
        Xu          Input data upper bounds (rows = objects, columns = features)
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
         
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            k = 0 # input pattern index
            while k < len(self.classId):
                idx_same_classes = (self.classId == self.classId[k]) | (self.classId == UNLABELED_CLASS) | ((self.classId != self.classId[k]) & (self.classId[k] == UNLABELED_CLASS))
                idx_same_classes[k] = 0 # remove element in the position k
                idex = torch.arange(len(self.classId))
                idex = idex[idx_same_classes] # keep the indices of elements retained
                V_same_class = self.V[idx_same_classes]
                W_same_class = self.W[idx_same_classes]
                
                if self.simil == 'short':
                    if isUsingGPU == False:
                        b = torch_memberG(self.W[k], self.V[k], V_same_class, W_same_class, self.gamma, self.oper, isUsingGPU)
                    else:
                        b = gpu_memberG(self.W[k], self.V[k], V_same_class, W_same_class, self.gamma, self.oper)
                        
                elif self.simil == 'long':
                    if isUsingGPU == False:
                        b = torch_memberG(self.V[k], self.W[k], V_same_class, W_same_class, self.gamma, self.oper, isUsingGPU)
                    else:
                        b = gpu_memberG(self.V[k], self.W[k], V_same_class, W_same_class, self.gamma, self.oper)
                        
                else:
                    b = torch_asym_similarity_one_many(self.V[k], self.W[k], V_same_class, W_same_class, self.gamma, self.sing, self.oper, isUsingGPU)
                
                sortB, indB = torch.sort(b, descending=True)
                idex = idex[indB]
                
                maxB = sortB[sortB >= self.bthres]	# apply membership threshold
                
                if len(maxB) > 0:
                    idexmax = idex[sortB >= self.bthres]
                    
                    if isUsingGPU == True:
                        idexmax = idexmax.cuda()
                        kMat = torch.cuda.LongTensor([k]).expand(idexmax.size(0))
                    else:
                        kMat = torch.LongTensor([k]).expand(idexmax.size(0))
                    
                    pairewise_maxb = torch.cat((torch.min(kMat, idexmax).reshape(-1, 1).float(), torch.max(kMat,idexmax).reshape(-1, 1).float(), maxB.reshape(-1, 1)), dim=1)

                    if isUsingGPU:
                        els = torch.arange(pairewise_maxb.size(0)).cuda()
                    else:
                        els = torch.arange(pairewise_maxb.size(0))
                        
                    for i in els:
                        # calculate new coordinates of k-th hyperbox by including pairewise_maxb(i,1)-th box, scrap the latter and leave the rest intact
                        # agglomorate pairewise_maxb(i, 0) and pairewise_maxb(i, 1) by adjusting pairewise_maxb(i, 0)
                        # remove pairewise_maxb(i, 1) by getting newV from 1 -> pairewise_maxb(i, 0) - 1, new coordinates for pairewise_maxb(i, 0), from pairewise_maxb(i, 0) + 1 -> pairewise_maxb(i, 1) - 1, pairewise_maxb(i, 1) + 1 -> end
                        row1 = pairewise_maxb[i, 0].long()
                        row2 = pairewise_maxb[i, 1].long()
                        
                        newV = torch.cat((self.V[:row1], torch.min(self.V[row1], self.V[row2]).reshape(1, -1), self.V[row1 + 1:row2], self.V[row2 + 1:]), dim=0)
                        newW = torch.cat((self.W[:row1], torch.max(self.W[row1], self.W[row2]).reshape(1, -1), self.W[row1 + 1:row2], self.W[row2 + 1:]), dim=0)
                        newClassId = torch.cat((self.classId[:row2], self.classId[row2 + 1:]))
                        if (newClassId[row1] == UNLABELED_CLASS):
                            newClassId[row1] = self.classId[row2]
                        # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                        # position of adjustment is pairewise_maxb[i, 0] in new bounds
                        if ((((newW[row1] - newV[row1]) <= self.teta).all() == True) and (not torch_modifiedIsOverlap(newV, newW, row1, newClassId, isUsingGPU))):
                            self.V = newV
                            self.W = newW
                            self.classId = newClassId
                                                   
                            isTraining = True
                            
                            if k != row1: # position pairewise_maxb[i, 1] (also k) is removed, so next step should start from pairewise_maxb[i, 1]
                                k = k - 1
                                
                            break # if hyperbox adjusted there's no need to look at other hyperboxes                            
                        
                k = k + 1
                    
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
    
    classifier = Torch_AccelBatchGFMM(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    print("Number of hyperboxes = ", len(classifier.classId))
    
    # Testing
    print("-- Testing --")
    result = classifier.predict_torch(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
        print("Training time = ", classifier.elapsed_training_time)