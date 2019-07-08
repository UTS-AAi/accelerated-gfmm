"""
Created on Wed Oct 31 17:42:53 2018

@author: Thanh Tung Khuat

Online GFMM classifier (training core) using Pytorch

    The membership grades are only computed on those hyperboxes. In constrast, in the previous version, we find membership grades for all current hyperboxes and then filter hyperboxes with the same label as the input pattern
    The normal version runs faster on the dataset with low dimensionality. The faster version runs well on the dataset with high dimensionality

     Torch_OnlineGFMM(gamma, teta, tMin, isDraw, oper, V, W, classId, isNorm, norm_range)
  
   INPUT
     V              Hyperbox lower bounds for the model to be updated using new data
     W              Hyperbox upper bounds for the model to be updated using new data
     classId        Hyperbox class labels (crisp)  for the model to be updated using new data
     gamma          Membership function slope (default: 1), datatype: array or scalar
     teta           Maximum hyperbox size (default: 1)
     tMin           Minimum value of Teta
     isDraw         Progress plot flag (default: False)
     oper           Membership calculation operation: 'min' or 'prod' (default: 'min')
     isNorm         Do normalization of input training samples or not?
     norm_range     New ranging of input data after normalization   
"""

import sys, os
sys.path.insert(0, os.path.pardir) 
import ast
import torch
torch.cuda.FloatTensor(1)
import numpy as np
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from functionhelper import is_Have_GPU, GPU_Computing_Threshold, UNLABELED_CLASS
from functionhelper.torch_membership_calc import torch_memberG, gpu_memberG
from functionhelper.torch_hyperboxadjustment import torch_hyperboxOverlapTest, torch_hyperboxContraction
from GFMM.classification import torch_predict
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from GFMM.torch_basegfmmclassifier import Torch_BaseGFMMClassifier

class Torch_OnlineGFMM(Torch_BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, tMin = 1, isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], V = torch.FloatTensor([]), W = torch.FloatTensor([]), classId = torch.LongTensor([])):
        Torch_BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.tMin = tMin
        self.V = V
        self.W = W
        self.classId = classId
        self.misclass = 1
        
        
    def fit(self, X_l, X_u, patClassId):
        """
        Training the classifier
        
         Xl             Input data lower bounds (rows = objects, columns = features)
         Xu             Input data upper bounds (rows = objects, columns = features)
         patClassId     Input data class labels (crisp). patClassId[i] = UNLABELED_CLASS corresponds to an unlabeled item
        
        """
        print('--Online Learning--')
        
        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
        
        if isinstance(X_l, torch.Tensor) == False:
            X_l = torch.from_numpy(X_l).float()
            X_u = torch.from_numpy(X_u).float()
            patClassId = torch.from_numpy(patClassId).long()
  
        time_start = time.perf_counter()
        
        yX, xX = X_l.size()
        teta = self.teta
        
        self.misclass = 1
        isUsingGPU = False
        
        while self.misclass > 0 and teta >= self.tMin:
            # for each input sample
            for i in range(yX):
                if len(self.V) > 0 and is_Have_GPU and isUsingGPU == False and self.V.size(0) * self.V.size(1) >= GPU_Computing_Threshold:
                    self.V = self.V.cuda()
                    self.W = self.W.cuda()
                    self.classId = self.classId.cuda()
                    isUsingGPU = True
                    
                if self.V.size(0) == 0:   # no model provided - starting from scratch
                    self.V = X_l[0].reshape(1, -1) # torch.DoubleTensor(X_l[0]).to(device)
                    self.W = X_u[0].reshape(1, -1) # torch.DoubleTensor(X_u[0]).to(device)
                    self.classId = torch.LongTensor([patClassId[0]]) # torch.DoubleTensor([patClassId[0]]).to(device)
                else:
                    if isUsingGPU == False:
                        classOfX = patClassId[i]
                    else:
                        classOfX = patClassId[i].cuda() 
                        
                    id_lb_sameX = (self.classId == classOfX) | (self.classId == UNLABELED_CLASS)
                    
                    if len(torch.nonzero(id_lb_sameX)) > 0: 
                        V_sameX = self.V[id_lb_sameX]
                        W_sameX = self.W[id_lb_sameX]
                        lb_sameX = self.classId[id_lb_sameX]
                        id_range = torch.arange(len(self.classId))
                        id_processing = id_range[id_lb_sameX]
                                         
                        if isUsingGPU == False:
                            Xl_cur = X_l[i]
                            Xu_cur = X_u[i]
                            
                            b = torch_memberG(Xl_cur, Xu_cur, V_sameX, W_sameX, self.gamma)
                        else:
                            Xl_cur = X_l[i].cuda()
                            Xu_cur = X_u[i].cuda()
                            
                            b = gpu_memberG(Xl_cur, Xu_cur, V_sameX, W_sameX, self.gamma)
                            
                        bSort, index = torch.sort(b, descending=True)
                        
                        if bSort[0] != 1 or (classOfX != lb_sameX[index[0]] and classOfX != UNLABELED_CLASS):
                            adjust = False
                            for j in id_processing[index]:
                                # test violation of max hyperbox size and class labels
                                if (classOfX == self.classId[j] or self.classId[j] == UNLABELED_CLASS or classOfX == UNLABELED_CLASS) and ((torch.max(self.W[j], Xu_cur).float() - torch.min(self.V[j], Xl_cur).float()) <= teta).all() == True:
                                    # adjust the j-th hyperbox
                                    self.V[j] = torch.min(self.V[j], Xl_cur)
                                    self.W[j] = torch.max(self.W[j], Xu_cur)
                                    indOfWinner = j
                                    adjust = True
                                    if classOfX != UNLABELED_CLASS and self.classId[j] == UNLABELED_CLASS:
                                        self.classId[j] = classOfX
                                        
                                    break
                                   
                            # if i-th sample did not fit into any existing box, create a new one
                            if not adjust:
                                self.V = torch.cat((self.V, Xl_cur.reshape(1, -1)), 0)
                                self.W = torch.cat((self.W, Xu_cur.reshape(1, -1)), 0)
                                if isUsingGPU == False:
                                    self.classId = torch.cat((self.classId, torch.LongTensor([classOfX])), 0)
                                else:
                                    self.classId = torch.cat((self.classId, torch.cuda.LongTensor([classOfX])), 0)
    
                            elif self.V.size(0) > 1:
                                for ii in range(self.V.size(0)):
                                    if ii != indOfWinner and self.classId[ii] != self.classId[indOfWinner]:
                                        caseDim = torch_hyperboxOverlapTest(self.V, self.W, indOfWinner, ii)		# overlap test
                                 
                                        if len(caseDim) > 0:
                                            self.V, self.W = torch_hyperboxContraction(self.V, self.W, caseDim, ii, indOfWinner)
                                            
                    else:
                        # create new sample
                        if isUsingGPU == False:
                            self.V = torch.cat((self.V, X_l[i].reshape(1, -1)), 0)
                            self.W = torch.cat((self.W, X_u[i].reshape(1, -1)), 0)
                            self.classId = torch.cat((self.classId, torch.LongTensor([patClassId[i]])), 0)
                        else:
                            self.V = torch.cat((self.V, X_l[i].cuda().reshape(1, -1)), 0)
                            self.W = torch.cat((self.W, X_l[i].cuda().reshape(1, -1)), 0)
                            self.classId = torch.cat((self.classId, torch.cuda.LongTensor([patClassId[i]])), 0)
    
    						
            teta = teta * 0.9
            if teta >= self.tMin:
                result = torch_predict(self.V, self.W, self.classId, X_l, X_u, patClassId, self.gamma, self.oper)
                self.misclass = result.summis

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
    arg6: + The minimum value of maximum size of hyperboxes (teta_min: default = teta)
    arg7: + gamma value (default: 1)
    arg8: operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg9: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg10: + range of input values after normalization (default: [0, 1])   
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
        teta_min = teta
    else:
        teta_min = float(sys.argv[6])
    
    if len(sys.argv) < 8:
        gamma = 1
    else:
        gamma = float(sys.argv[7])
    
    if len(sys.argv) < 9:
        oper = 'min'
    else:
        oper = sys.argv[8]
    
    if len(sys.argv) < 10:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[9])
    
    if len(sys.argv) < 11:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[10])
    
    # print('isDraw = ', isDraw, ' teta = ', teta, ' teta_min = ', teta_min, ' gamma = ', gamma, ' oper = ', oper, ' isNorm = ', isNorm, ' norm_range = ', norm_range)
    
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
        
    classifier = Torch_OnlineGFMM(gamma, teta, teta_min, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict_torch(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
        print("Training time = ", classifier.elapsed_training_time)
        

