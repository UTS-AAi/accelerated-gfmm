# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:39:47 2018

@author: Thanh Tung Khuat

onlnGFMM - Online GFMM classifier (training core). Faster version on datasets with high dimensionality

    Compare to previous version, in this coding, we find the hyperboxes representing the same label as the input pattern,
    The membership grades are only computed on those hyperboxes. In constrast, in the previous version, we find membership grades for all current hyperboxes and then filter hyperboxes with the same label as the input pattern
    The normal version runs faster on the dataset with low dimensionality. The faster version runs well on the dataset with high dimensionality

     OnlineGFMM(gamma, teta, tMin, isDraw, oper, V, W, classId, isNorm, norm_range)

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
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)

import ast
import numpy as np
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from functionhelper import UNLABELED_CLASS
from functionhelper.membershipcalc import memberG
from functionhelper.hyperboxadjustment import hyperboxOverlapTest, hyperboxContraction
from GFMM.classification import predict
from functionhelper.drawinghelper import drawbox
from functionhelper.preprocessinghelper import loadDataset, string_to_boolean
from GFMM.basegfmmclassifier import BaseGFMMClassifier

class OnlineGFMM(BaseGFMMClassifier):

    def __init__(self, gamma = 1, teta = 1, tMin = 1, isDraw = False, oper = 'min', isNorm = False, norm_range = [0, 1], V = np.array([]), W = np.array([]), classId = np.array([])):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)

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
         patClassId     Input data class labels (crisp). patClassId[i] = 0 corresponds to an unlabeled item

        """
        print('--Online Learning--')

        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
        #X_l = X_l.astype(np.float32)
        #X_u = X_u.astype(np.float32)
        time_start = time.perf_counter()

        yX, xX = X_l.shape
        teta = self.teta

        mark = np.array(['*', 'o', 'x', '+', '.', ',', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_'])
        mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])

        listLines = list()
        listInputSamplePoints = list()

        if self.isDraw:
            drawing_canvas = self.initializeCanvasGraph("GFMM - Online learning", xX)

            if self.V.size > 0:
                # draw existed hyperboxes
                color_ = np.array(['k'] * len(self.classId), dtype = object)
                for c in range(len(self.classId)):
                    if self.classId[c] < len(mark_col):
                        color_[c] = mark_col[self.classId[c]]

                hyperboxes = drawbox(self.V[:, 0:np.minimum(xX,3)], self.W[:, 0:np.minimum(xX,3)], drawing_canvas, color_)
                listLines.extend(hyperboxes)
                self.delay()

        self.misclass = 1

        while self.misclass > 0 and teta >= self.tMin:
            # for each input sample
            for i in range(yX):
                classOfX = patClassId[i]
                # draw input samples
                if self.isDraw:
                    if i == 0 and len(listInputSamplePoints) > 0:
                        # reset input point drawing
                        for point in listInputSamplePoints:
                            point.remove()
                        listInputSamplePoints.clear()

                    color_ = 'k'
                    if classOfX < len(mark_col):
                        color_ = mark_col[classOfX]

                    if (X_l[i, :] == X_u[i, :]).all():
                        marker_ = 'd'
                        if classOfX < len(mark):
                            marker_ = mark[classOfX]

                        if xX == 2:
                            inputPoint = drawing_canvas.plot(X_l[i, 0], X_l[i, 1], color = color_, marker=marker_)
                        else:
                            inputPoint = drawing_canvas.plot([X_l[i, 0]], [X_l[i, 1]], [X_l[i, 2]], color = color_, marker=marker_)

                        #listInputSamplePoints.append(inputPoint)
                    else:
                        inputPoint = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, color_)

                    listInputSamplePoints.append(inputPoint[0])
                    self.delay()

                if self.V.size == 0:   # no model provided - starting from scratch
                    self.V = np.array([X_l[0]])
                    self.W = np.array([X_u[0]])
                    self.classId = np.array([patClassId[0]])

                    if self.isDraw == True:
                        # draw hyperbox
                        box_color = 'k'
                        if patClassId[0] < len(mark_col):
                            box_color = mark_col[patClassId[0]]

                        hyperbox = drawbox(np.asmatrix(self.V[0, 0:np.minimum(xX,3)]), np.asmatrix(self.W[0, 0:np.minimum(xX,3)]), drawing_canvas, box_color)
                        listLines.append(hyperbox[0])
                        self.delay()

                else:
                    id_lb_sameX = (self.classId == classOfX) | (self.classId == UNLABELED_CLASS)
                    
                    if id_lb_sameX.any() == True: 
                        # if we have small number of hyperboxes with low dimension, this operation takes more time compared to computing membership value with all hyperboxes and ignore
                        # hyperboxes with different class (the membership computation on small dimensionality is so rapidly). However, if we have hyperboxes with high dimensionality, the membership computing on all hyperboxes take so long => reduced to only hyperboxes with the
                        # same class will significantly decrease the running time
                        V_sameX = self.V[id_lb_sameX]
                        W_sameX = self.W[id_lb_sameX]
                        lb_sameX = self.classId[id_lb_sameX]
                        id_range = np.arange(len(self.classId))
                        id_processing = id_range[id_lb_sameX]
    
                        b = memberG(X_l[i], X_u[i], V_sameX, W_sameX, self.gamma)
                        index = np.argsort(b)[::-1]
                        bSort = b[index]
                    
                        if bSort[0] != 1 or (classOfX != lb_sameX[index[0]] and classOfX != UNLABELED_CLASS):
                            adjust = False
                            for j in id_processing[index]:
                                # test violation of max hyperbox size and class labels
                                # This code can be accelerated by assigning the np.minimum(self.V[j], X_l[i]) and np.maximum(self.W[j], X_u[i]) to temporal variables so that they are not computed twice
                                if (classOfX == self.classId[j] or self.classId[j] == UNLABELED_CLASS or classOfX == UNLABELED_CLASS) and ((np.maximum(self.W[j], X_u[i]) - np.minimum(self.V[j], X_l[i])) <= teta).all() == True:
                                    # adjust the j-th hyperbox
                                    self.V[j] = np.minimum(self.V[j], X_l[i])
                                    self.W[j] = np.maximum(self.W[j], X_u[i])
                                    indOfWinner = j
                                    adjust = True
                                    if classOfX != UNLABELED_CLASS and self.classId[j] == UNLABELED_CLASS:
                                        self.classId[j] = classOfX
    
                                    if self.isDraw:
                                        # Handle drawing graph
                                        box_color = 'k'
                                        if self.classId[j] < len(mark_col):
                                            box_color = mark_col[self.classId[j]]
    
                                        try:
                                            listLines[j].remove()
                                        except:
                                            pass
    
                                        hyperbox = drawbox(np.asmatrix(self.V[j, 0:np.minimum(xX, 3)]), np.asmatrix(self.W[j, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                        listLines[j] = hyperbox[0]
                                        self.delay()
    
                                    break

                            # if i-th sample did not fit into any existing box, create a new one
                            if not adjust:
                                self.V = np.concatenate((self.V, X_l[i].reshape(1, -1)), axis = 0)
                                self.W = np.concatenate((self.W, X_u[i].reshape(1, -1)), axis = 0)
                                self.classId = np.concatenate((self.classId, [classOfX]))
        
                                if self.isDraw:
                                    # handle drawing graph
                                    box_color = 'k'
                                    if self.classId[-1] < len(mark_col):
                                        box_color = mark_col[self.classId[-1]]
        
                                    hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                                    listLines.append(hyperbox[0])
                                    self.delay()
        
                            elif self.V.shape[0] > 1:
                                for ii in range(self.V.shape[0]):
                                    if ii != indOfWinner and (self.classId[ii] != self.classId[indOfWinner] or self.classId[indOfWinner] == UNLABELED_CLASS):
                                        caseDim = hyperboxOverlapTest(self.V, self.W, indOfWinner, ii)		# overlap test
        
                                        if caseDim.size > 0:
                                            self.V, self.W = hyperboxContraction(self.V, self.W, caseDim, ii, indOfWinner)
                                            if self.isDraw:
                                                # Handle graph drawing
                                                boxii_color = boxwin_color = 'k'
                                                if self.classId[ii] < len(mark_col):
                                                    boxii_color = mark_col[self.classId[ii]]
        
                                                if self.classId[indOfWinner] < len(mark_col):
                                                    boxwin_color = mark_col[self.classId[indOfWinner]]
        
                                                try:
                                                    listLines[ii].remove()
                                                    listLines[indOfWinner].remove()
                                                except:
                                                    pass
        
                                                hyperboxes = drawbox(self.V[[ii, indOfWinner], 0:np.minimum(xX, 3)], self.W[[ii, indOfWinner], 0:np.minimum(xX, 3)], drawing_canvas, [boxii_color, boxwin_color])
                                                listLines[ii] = hyperboxes[0]
                                                listLines[indOfWinner] = hyperboxes[1]
                                                self.delay()

                    else:
                        self.V = np.concatenate((self.V, X_l[i].reshape(1, -1)), axis = 0)
                        self.W = np.concatenate((self.W, X_u[i].reshape(1, -1)), axis = 0)
                        self.classId = np.concatenate((self.classId, [classOfX]))

                        if self.isDraw:
                            # handle drawing graph
                            box_color = 'k'
                            if self.classId[-1] < len(mark_col):
                                box_color = mark_col[self.classId[-1]]

                            hyperbox = drawbox(np.asmatrix(X_l[i, 0:np.minimum(xX, 3)]), np.asmatrix(X_u[i, 0:np.minimum(xX, 3)]), drawing_canvas, box_color)
                            listLines.append(hyperbox[0])
                            self.delay()
                            
            teta = teta * 0.9
            if teta >= self.tMin:
                result = predict(self.V, self.W, self.classId, X_l, X_u, patClassId, self.gamma, self.oper)
                self.misclass = result.summis

        # Draw last result
#        if self.isDraw == True:
#            # Handle drawing graph
#            drawing_canvas.cla()
#            color_ = np.empty(len(self.classId), dtype = object)
#            for c in range(len(self.classId)):
#                color_[c] = mark_col[self.classId[c]]
#
#            drawbox(self.V[:, 0:np.minimum(xX, 3)], self.W[:, 0:np.minimum(xX, 3)], drawing_canvas, color_)
#            self.delay()
#
#        if self.isDraw:
#            plt.show()

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
    arg4: + path to file containing the validation dataset
    arg5: + True: drawing hyperboxes during the training process
          + False: no drawing
    arg6: + Maximum size of hyperboxes (teta, default: 1)
    arg7: + The minimum value of maximum size of hyperboxes (teta_min: default = teta)
    arg8: + gamma value (default: 1)
    arg9: operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg10: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg11: + range of input values after normalization (default: [0, 1])
    """
    # Init default parameters
    if len(sys.argv) < 6:
        isDraw = False
    else:
        isDraw = string_to_boolean(sys.argv[5])

    if len(sys.argv) < 7:
        teta = 1
    else:
        teta = float(sys.argv[6])

    if len(sys.argv) < 8:
        teta_min = teta
    else:
        teta_min = float(sys.argv[7])

    if len(sys.argv) < 9:
        gamma = 1
    else:
        gamma = float(sys.argv[8])

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

    # print('isDraw = ', isDraw, ' teta = ', teta, ' teta_min = ', teta_min, ' gamma = ', gamma, ' oper = ', oper, ' isNorm = ', isNorm, ' norm_range = ', norm_range)
    start_t = time.perf_counter()
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
    
    validation_file = sys.argv[4]
    
    if (not validation_file) == True:
        # empty validation file
        print('no pruning')
        isPruning = False
    else:
        print('pruning')
        isPruning = True
        Xval, _, patClassIdVal, _ = loadDataset(validation_file, 1, False)
    
    classifier = OnlineGFMM(gamma, teta, teta_min, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    print("Before pruning:")
    print('V size = ', classifier.V.shape)
    print('W size = ', classifier.W.shape)
    if isPruning == True:
        classifier.pruning_val(Xval, Xval, patClassIdVal)
        print("After pruning:")
        print('V size = ', classifier.V.shape)
        print('W size = ', classifier.W.shape)
    end_t = time.perf_counter()
    
    print("Reading file + Training and pruning Time = ", end_t - start_t)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")

    print("-- Testing on Validation file --")
    result = classifier.predict(Xval, patClassIdVal)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xval.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")