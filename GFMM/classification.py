# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:22:08 2018

@author: Thanh Tung Khuat

GFMM Predictor

"""

import numpy as np
import torch
from functionhelper.membershipcalc import memberG
from functionhelper.bunchdatatype import Bunch
from functionhelper.torch_membership_calc import torch_memberG, gpu_memberG
from functionhelper import device, GPU_Computing_Threshold, is_Have_GPU, UNLABELED_CLASS

#def predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
#    """
#    GFMM classifier (test routine)
#
#      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)
#
#    INPUT
#      V                 Tested model hyperbox lower bounds
#      W                 Tested model hyperbox upper bounds
#      classId	          Input data (hyperbox) class labels (crisp)
#      XlT               Test data lower bounds (rows = objects, columns = features)
#      XuT               Test data upper bounds (rows = objects, columns = features)
#      patClassIdTest    Test data class labels (crisp)
#      gama              Membership function slope (default: 1)
#      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')
#
#   OUTPUT
#      result           A object with Bunch datatype containing all results as follows:
#                          + summis           Number of misclassified objects
#                          + misclass         Binary error map
#                          + sumamb           Number of objects with maximum membership in more than one class
#                          + out              Soft class memberships
#                          + mem              Hyperbox memberships
#
#    """
#
#    #initialization
#    yX = XlT.shape[0]
#    misclass = np.zeros(yX)
#    classes = np.unique(classId)
#    noClasses = classes.size
#    ambiguity = np.zeros((yX, 1))
#    mem = np.zeros((yX, V.shape[0]))
#    out = np.zeros((yX, noClasses))
#
#    # classifications
#    for i in range(yX):
#        mem[i, :] = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
#        bmax = mem[i,:].max()	                                          # get max membership value
#        maxVind = np.nonzero(mem[i,:] == bmax)[0]                         # get indexes of all hyperboxes with max membership
#
#        for j in range(noClasses):
#            out[i, j] = mem[i, classId == classes[j]].max()            # get max memberships for each class
#
#        ambiguity[i, :] = np.sum(out[i, :] == bmax) 						  # number of different classes with max membership
#
#        if bmax == 0:
#            print('zero maximum membership value')                     # this is probably bad...
#            misclass[i] = True
#        else:
##        misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
#            if len(np.unique(classId[maxVind])) > 1:
#                misclass[i] = True
#            else:
#                misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
#                
#    # results
#    sumamb = np.sum(ambiguity[:, 0] > 1)
#    summis = np.sum(misclass).astype(np.int64)
#
#    result = Bunch(summis = summis, misclass = misclass, sumamb = sumamb, out = out, mem = mem)
#    return result


#def torch_predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
#    """
#    GFMM classifier (test routine). Implemented by Pytorch
#
#      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)
#
#    INPUT
#      V                 Tested model hyperbox lower bounds
#      W                 Tested model hyperbox upper bounds
#      classId	          Input data (hyperbox) class labels (crisp)
#      XlT               Test data lower bounds (rows = objects, columns = features)
#      XuT               Test data upper bounds (rows = objects, columns = features)
#      patClassIdTest    Test data class labels (crisp)
#      gama              Membership function slope (default: 1)
#      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')
#
#   OUTPUT
#      result           A object with Bunch datatype containing all results as follows:
#                          + summis           Number of misclassified objects
#                          + misclass         Binary error map
#                          + sumamb           Number of objects with maximum membership in more than one class
#                          + out              Soft class memberships
#                          + mem              Hyperbox memberships
#
#    """
#    #initialization
#    yX = XlT.size(0)
#    isUsingGPU = False
#    if is_Have_GPU and (W.size(0) * W.size(1) >= GPU_Computing_Threshold or XlT.size(1) >= GPU_Computing_Threshold):
#        V = V.cuda()
#        W = W.cuda()
#        classId = classId.cuda()
#        XlT = XlT.cuda()
#        XuT = XuT.cuda()
#        patClassIdTest = patClassIdTest.cuda()
#        misclass = torch.cuda.FloatTensor(yX).fill_(0)
#        classes = torch.unique(classId)
#        noClasses = classes.size(0)
#        ambiguity = torch.cuda.FloatTensor(yX, 1).fill_(0)
#        mem = torch.cuda.FloatTensor(yX, V.size(0)).fill_(0)
#        out = torch.cuda.FloatTensor(yX, noClasses).fill_(0)
#        isUsingGPU = True
#        els = torch.arange(yX).cuda()
#    else:
#        classes = torch.unique(classId)
#        misclass = torch.zeros(yX)
#        noClasses = classes.size(0)
#        ambiguity = torch.zeros((yX, 1))
#        mem = torch.zeros((yX, V.size(0)))
#        out = torch.zeros((yX, noClasses))
#        els = torch.arange(yX)
#
#    # classifications
#    for i in els:
#        if isUsingGPU == True:
#            mem[i, :] = gpu_memberG(XlT[i, :], XuT[i, :], V, W, gama, oper)
#        else:
#            mem[i, :] = torch_memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
#
#        bmax = mem[i,:].max()	                                          # get max membership value
#        maxVind = torch.nonzero(mem[i,:] == bmax)                         # get indexes of all hyperboxes with max membership
#
#        for j in torch.arange(noClasses):
#            out[i, j] = mem[i, classId == classes[j]].max()            # get max memberships for each class
#
#        ambiguity[i, :] = torch.sum(out[i, :] == bmax) 						  # number of different classes with max membership
#
#        if bmax == 0:
#            print('zero maximum membership value')                     # this is probably bad...
#
#        misclass[i] = ~(torch.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
#
#    # results
#    sumamb = torch.sum(ambiguity[:, 0] > 1)
#    summis = torch.sum(misclass)
#
#    result = Bunch(summis = summis, misclass = misclass, sumamb = sumamb, out = out, mem = mem)
#    return result

def predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine)

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships

    """

    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)

    # classifications
    for i in range(yX):
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

        if bmax == 0:
            print('zero maximum membership value')                     # this is probably bad...
            misclass[i] = True
        else:
            if len(np.unique(classId[maxVind])) > 1:
                misclass[i] = True
            else:
                if np.any(classId[maxVind] == patClassIdTest[i]) == True or patClassIdTest[i] == UNLABELED_CLASS:
                    misclass[i] = False
                else:
                    misclass[i] = True
                #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass)
    return result

  
def torch_predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine). Implemented by Pytorch

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships

    """
    #initialization
    yX = XlT.size(0)
    isUsingGPU = False
    if is_Have_GPU and (W.size(0) * W.size(1) >= GPU_Computing_Threshold or XlT.size(1) >= GPU_Computing_Threshold):
        V = V.cuda()
        W = W.cuda()
        classId = classId.cuda()
        XlT = XlT.cuda()
        XuT = XuT.cuda()
        patClassIdTest = patClassIdTest.cuda()
        misclass = torch.cuda.FloatTensor(yX).fill_(0)
        isUsingGPU = True
        els = torch.arange(yX).cuda()
    else:
        misclass = torch.zeros(yX)
        els = torch.arange(yX)

    # classifications
    for i in els:
        if isUsingGPU == True:
            mem = gpu_memberG(XlT[i, :], XuT[i, :], V, W, gama, oper)
        else:
            mem = torch_memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes

        bmax = mem.max()	                                          # get max membership value
        maxVind = mem == bmax                        # get indexes of all hyperboxes with max membership

        if bmax == 0:
            print('zero maximum membership value')                     # this is probably bad...
            misclass[i] = 1
        else:
            if len(torch.unique(classId[maxVind])) > 1:
                misclass[i] = 1
            else:
                if (torch.any(classId[maxVind] == patClassIdTest[i]) == 1) or (patClassIdTest[i] == UNLABELED_CLASS):
                    misclass[i] = 0
                else:
                    misclass[i] = 1

    # results
    summis = torch.sum(misclass)

    result = Bunch(summis = summis, misclass = misclass)
    return result


def predictDecisionLevelEnsemble(classifiers, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    Perform classification for a decision level ensemble learning

                result = predictDecisionLevelEnsemble(classifiers, XlT, XuT, patClassIdTest, gama, oper)

    INPUT
        classifiers         An array of classifiers needed to combine, datatype of each element in the array is BaseGFMMClassifier
        XlT                 Test data lower bounds (rows = objects, columns = features)
        XuT                 Test data upper bounds (rows = objects, columns = features)
        patClassIdTest      Test data class labels (crisp)
        gama                Membership function slope (default: 1)
        oper                Membership calculation operation: 'min' or 'prod' (default: 'min')

    OUTPUT
        result              A object with Bunch datatype containing all results as follows:
                                + summis        Number of misclassified samples
                                + misclass      Binary error map for input samples
                                + out           Soft class memberships, rows are testing input patterns, columns are indices of classes
                                + classes       Store class labels corresponding column indices of out
    """
    numClassifier = len(classifiers)

    yX = XlT.shape[0]
    misclass = np.zeros(yX, dtype=np.bool)
    # get all class labels of all base classifiers
    classId = classifiers[0].classId
    for i in range(numClassifier):
        if i != 0:
            classId = np.union1d(classId, classifiers[i].classId)

    classes = np.unique(classId)
    noClasses = len(classes)
    out = np.zeros((yX, noClasses), dtype=np.float64)

    # classification of each testing pattern i
    for i in range(yX):
        for idClf in range(numClassifier):
            # calculate memberships for all hyperboxes of classifier idClf
            mem_tmp = memberG(XlT[i, :], XuT[i, :], classifiers[idClf].V, classifiers[idClf].W, gama, oper)

            for j in range(noClasses):
                # get max membership of hyperboxes with class label j
                same_j_labels = mem_tmp[classifiers[idClf].classId == classes[j]]
                if len(same_j_labels) > 0:
                    mem_max = same_j_labels.max()
                    out[i, j] = out[i, j] + mem_max

        # compute membership value of each class over all classifiers
        out[i, :] = out[i, :] / numClassifier
        # get max membership value for each class with regard to the i-th sample
        maxb = out[i].max()
        # get positions of indices of all classes with max membership
        maxMemInd = out[i] == maxb
        #misclass[i] = ~(np.any(classes[maxMemInd] == patClassIdTest[i]) | (patClassIdTest[i] == 0))
        misclass[i] = np.logical_or((classes[maxMemInd] == patClassIdTest[i]).any(), patClassIdTest[i] == UNLABELED_CLASS) != True

    # count number of missclassified patterns
    summis = np.sum(misclass)

    result = Bunch(summis = summis, misclass = misclass, out = out, classes = classes)
    return result


def predictOnlineOfflineCombination(onlClassifier, offClassifier, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM online-offline classifier (test routine)

      result = predictOnlineOfflineCombination(onlClassifier, offClassifier, XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      onlClassifier   online classifier with the following attributes:
                        + V: hyperbox lower bounds
                        + W: hyperbox upper bounds
                        + classId: hyperbox class labels (crisp)

      offClassifier   offline classifier with the following attributes:
                        + V: hyperbox lower bounds
                        + W: hyperbox upper bounds
                        + classId: hyperbox class labels (crisp)

      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + out              Soft class memberships

    """

    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    classes = np.union1d(onlClassifier.classId, offClassifier.classId)
    noClasses = classes.size
    mem_onl = np.zeros((yX, onlClassifier.V.shape[0]))
    mem_off = np.zeros((yX, offClassifier.V.shape[0]))
    out = np.zeros((yX, noClasses))

    # classifications
    for i in range(yX):
        mem_onl[i, :] = memberG(XlT[i, :], XuT[i, :], onlClassifier.V, onlClassifier.W, gama, oper) # calculate memberships for all hyperboxes in the online classifier
        bmax_onl = mem_onl[i, :].max()	                                   # get max membership value among hyperboxes in the online classifier
        maxVind_onl = np.nonzero(mem_onl[i,:] == bmax_onl)[0]             # get indexes of all hyperboxes in the online classifier with max membership

        mem_off[i, :] = memberG(XlT[i, :], XuT[i, :], offClassifier.V, offClassifier.W, gama, oper) # calculate memberships for all hyperboxes in the offline classifier
        bmax_off = mem_off[i, :].max()	                                   # get max membership value among hyperboxes in the offline classifier
        maxVind_off = np.nonzero(mem_off[i,:] == bmax_off)[0]                 # get indexes of all hyperboxes in the offline classifier with max membership


        for j in range(noClasses):
            out_onl_mems = mem_onl[i, onlClassifier.classId == classes[j]]            # get max memberships for each class of online classifier
            if len(out_onl_mems) > 0:
                out_onl = out_onl_mems.max()
            else:
                out_onl = 0

            out_off_mems = mem_off[i, offClassifier.classId == classes[j]]            # get max memberships for each class of offline classifier
            if len(out_off_mems) > 0:
                out_off = out_off_mems.max()
            else:
                out_off = 0

            if out_onl > out_off:
                out[i, j] = out_onl
            else:
                out[i, j] = out_off

        if bmax_onl > bmax_off:
            if len(np.unique(onlClassifier.classId[maxVind_onl])) > 1:
                if len(np.unique(offClassifier.classId[maxVind_off])) > 1:
                    misclass[i] = True
                else:
                    misclass[i] = ~(np.any(offClassifier.classId[maxVind_off] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
            else:
                misclass[i] = ~(np.any(onlClassifier.classId[maxVind_onl] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
        else:
            if len(np.unique(offClassifier.classId[maxVind_off])) > 1:
                if len(np.unique(onlClassifier.classId[maxVind_onl])) > 1:
                    misclass[i] = True
                else:
                    misclass[i] = ~(np.any(onlClassifier.classId[maxVind_onl] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))
            else:
                misclass[i] = ~(np.any(offClassifier.classId[maxVind_off] == patClassIdTest[i]) | (patClassIdTest[i] == UNLABELED_CLASS))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, out = out)
    return result
