# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 08:44:50 2018

@author: Thanh Tung Khuat

Hyperbox adjustment handling: overlap testing, hyperbox contraction

Implemented by Pytorch

"""

import torch
from functionhelper import GPU_Computing_Threshold, is_Have_GPU

def torch_hyperboxOverlapTest(V, W, ind, testInd):
    """
    Hyperbox overlap test

      dim = overlapTest(V, W, ind, testInd)
  
    INPUT
        V           Hyperbox lower bounds (datatype: FloatTensor)
        W           Hyperbox upper bounds (datatype: FloatTensor)
        ind         Index of extended hyperbox
        testInd     Index of hyperbox to test for overlap with the extended hyperbox

    OUTPUT
        dim         Result to be fed into contrG1, which is special numpy array

    """
    Va = V[ind, :]
    Vb = V[testInd, :]
    Wa = W[ind, :]
    Wb = W[testInd, :]
    
    if is_Have_GPU == True and Va.size(0) >= GPU_Computing_Threshold:
        Va = Va.cuda()
        Vb = Vb.cuda()
        Wa = Wa.cuda()
        Wb = Wb.cuda()
    
    dim = []
    xW = W.size(1)
    
    condWiWk = Wa - Wb > 0
    condViVk = Va - Vb > 0
    condWkVi = Wb - Va > 0
    condWiVk = Wa - Vb > 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condWiWk & ~condViVk
    c4 = ~condWiWk & condViVk
    c = c1 + c2 + c3 + c4

    ad = c.all()

    if ad == True:
        minimum = 1;
        # Because of only one element, using CPU to compute
        for i in range(xW):
            if c1[i] == True:
                if minimum > W[ind, i] - V[testInd, i]:
                    minimum = W[ind, i] - V[testInd, i]
                    dim = [1, i]
            
            elif c2[i] == True:
                if minimum > W[testInd, i] - V[ind, i]:
                    minimum = W[testInd, i] - V[ind, i]
                    dim = [2, i]
            
            elif c3[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = [31, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = [32, i]
                    
            elif c4[i] == True:
                if minimum > (W[testInd, i] - V[ind, i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = [41, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = [42, i]
                
    return dim

def torch_hyperboxContraction(V1, W1, newCD, testedInd, ind):
    """
    Adjusting min-max points of overlaping clusters (with meet halfway)

      V, W = hyperboxContraction(V,W,newCD,testedInd,ind)
  
    INPUT
      V1            Lower bounds of existing hyperboxes
      W1            Upper bounds of existing hyperboxes
      newCD         Special parameters, output from hyperboxOverlapTest
      testedInd     Index of hyperbox to test for overlap with the extended hyperbox
      ind           Index of extended hyperbox	
   
    OUTPUT
      V             Lower bounds of adjusted hyperboxes
      W             Upper bounds of adjusted hyperboxes
    
    """
    V = V1
    W = W1
    
    if newCD[0] == 1:
        W[ind, newCD[1]] = (V[testedInd, newCD[1]] + W[ind, newCD[1]]) / 2
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    elif newCD[0] == 2:
        V[ind, newCD[1]] = (W[testedInd, newCD[1]] + V[ind, newCD[1]]) / 2
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
    elif newCD[0] == 31:
        V[ind, newCD[1]] = W[testedInd, newCD[1]]
    elif newCD[0] == 32:
        W[ind, newCD[1]] = V[testedInd, newCD[1]]
    elif newCD[0] == 41:
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
    elif newCD[0] == 42:
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    
    return (V, W)


def torch_isOverlap(V, W, ind, classId, isUsingGPU = False):
    """
    Checking overlap between hyperbox ind and remaning hyperboxes (1 vs many)
    
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of the hyperbox to be checked for overlap
        classId     Class labels of hyperboxes
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    
    isUsingGPULocal = isUsingGPU
    if isUsingGPU == False and is_Have_GPU == True and W.shape[0] * W.shape[1] >= GPU_Computing_Threshold:
        V = V.cuda()
        W = W.cuda()
        isUsingGPULocal = True
    
    if (V[ind] > W[ind]).any() == True:
        return False
    else:
        indcomp = torch.nonzero((W >= V).all(dim = 1)) 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(indcomp) == 0:
            return False
        else:
            newInd = indcomp[indcomp != ind]
            
            if len(newInd) > 0:
                if isUsingGPULocal == False:
                    onesTemp = torch.ones((len(newInd), 1))
                else:
                    onesTemp = torch.cuda.FloatTensor(len(newInd), 1).fill_(1)
                    
                condWiWk = (onesTemp * W[ind] - W[newInd]) > 0
                condViVk = (onesTemp * V[ind] - V[newInd]) > 0
                condWkVi = (W[newInd] - onesTemp * V[ind]) > 0
                condWiVk = (onesTemp * W[ind] - V[newInd]) > 0
                
                #print(condWiWk.shape)
                
                c1 = ~condWiWk & ~condViVk & condWiVk
                c2 = condWiWk & condViVk & condWkVi
                c3 = condWiWk & ~condViVk
                c4 = ~condWiWk & condViVk
                
                c = c1 + c2 + c3 + c4
                
                ad = c.all(dim = 1)
                #print("Ad = ", np.nonzero(ad)[0].size)
                ind2 = newInd[ad]
                
                ovresult = (classId[ind2] != classId[ind]).any()
                
                if isUsingGPULocal == True and isUsingGPU == False:
                    ovresult = ovresult.cpu()
                
                return ovresult
            else:
                return False

def torch_modifiedIsOverlap(V, W, ind, classId, isUsingGPU = False):
    """
    Checking overlap between hyperbox ind and remaning hyperboxes (1 vs many)
    Only do overlap testing with hyperboxes belonging to other classes
    
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of the hyperbox to be checked for overlap
        classId     Class labels of hyperboxes
        
    OUTPUT
        False - no overlap,  True - overlap
    """
    isUsingGPULocal = isUsingGPU
    if isUsingGPU == False and is_Have_GPU == True and W.shape[0] * W.shape[1] >= GPU_Computing_Threshold:
        V = V.cuda()
        W = W.cuda()
        isUsingGPULocal = True
        
    if (V[ind] > W[ind]).any() == True:
        return False
    else:
        indcomp = torch.nonzero((W >= V).all(dim = 1)) 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(indcomp) == 0:
            return False
        else:
            class_indcomp = classId[indcomp]
            newInd = indcomp[class_indcomp != classId[ind]] # get index of hyperbox representing different classes
            
            if len(newInd) > 0:
                if isUsingGPULocal == False:
                    onesTemp = torch.ones((len(newInd), 1))
                else:
                    onesTemp = torch.cuda.FloatTensor(len(newInd), 1).fill_(1)
                    
                condWiWk = (onesTemp * W[ind] - W[newInd]) > 0
                condViVk = (onesTemp * V[ind] - V[newInd]) > 0
                condWkVi = (W[newInd] - onesTemp * V[ind]) > 0
                condWiVk = (onesTemp * W[ind] - V[newInd]) > 0
                
                #print(condWiWk.shape)
                
                c1 = ~condWiWk & ~condViVk & condWiVk
                c2 = condWiWk & condViVk & condWkVi
                c3 = condWiWk & ~condViVk
                c4 = ~condWiWk & condViVk
                
                c = c1 + c2 + c3 + c4
                
                ad = c.all(dim = 1)
                #print("Ad = ", np.nonzero(ad)[0].size)
                ind2 = newInd[ad]
                
                ovresult = (classId[ind2] != classId[ind]).any()
                
                if isUsingGPULocal == True and isUsingGPU == False:
                    ovresult = ovresult.cpu()
                    
                return ovresult
            else:
                return False
               
            
def torch_improvedHyperboxOverlapTest(V, W, ind, testInd, Xh):
    """
    Hyperbox overlap test - 9 cases

      dim = overlapTest(V, W, ind, testInd)
  
    INPUT
        V           Hyperbox lower bounds
        W           Hyperbox upper bounds
        ind         Index of extended hyperbox
        testInd     Index of hyperbox to test for overlap with the extended hyperbox
        Xh          Current input sample being considered (used for case 9)

    OUTPUT
        dim         Result to be fed into contrG1, which is special numpy array

    """
    dim = []
    xW = W.size(1)
    
    Va = V[ind, :]
    Vb = V[testInd, :]
    Wa = W[ind, :]
    Wb = W[testInd, :]
    
    if is_Have_GPU == True and Va.size(0) >= GPU_Computing_Threshold:
        Va = Va.cuda()
        Vb = Vb.cuda()
        Wa = Wa.cuda()
        Wb = Wb.cuda()
    
    condWiWk = Wa - Wb > 0
    condViVk = Va - Vb > 0
    condWkVi = Wb - Va > 0
    condWiVk = Wa - Vb > 0
    
    condEqViVk = Va - Vb == 0
    condEqWiWk = Wa - Wb == 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condEqViVk & condWiVk & ~condWiWk
    c4 = ~condViVk & condWiVk & condEqWiWk
    c5 = condEqViVk & condWkVi & condWiWk
    c6 = condViVk & condWkVi & condEqWiWk
    c7 = ~condViVk & condWiWk
    c8 = condViVk & ~condWiWk
    c9 = condEqViVk & ~condViVk & condEqWiWk
    
    c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9

    ad = c.all()

    if ad == True:
        minimum = 1
        for i in range(xW):
            if c1[i] == True:
                if minimum > W[ind, i] - V[testInd, i]:
                    minimum = W[ind, i] - V[testInd, i]
                    dim = [1, i]
            
            elif c2[i] == True:
                if minimum > W[testInd, i] - V[ind, i]:
                    minimum = W[testInd, i] - V[ind, i]
                    dim = [2, i]
            
            elif c3[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                
                dim = [3, i]
                    
            elif c4[i] == True:
                if minimum > (W[testInd, i] - V[ind, i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                
                dim = [4, i]
                
            elif c5[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    
                dim = [5, i]
            
            elif c6[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    
                dim = [6, i]
                
            elif c7[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = [71, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = [72, i]
                    
            elif c8[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]) and (W[testInd, i] - V[ind, i]) < (W[ind, i] - V[testInd, i]):
                    minimum = W[testInd, i] - V[ind, i]
                    dim = [81, i]
                elif minimum > (W[ind, i] - V[testInd, i]):
                    minimum = W[ind, i] - V[testInd, i]
                    dim = [82, i]
                    
            elif c9[i] == True:
                if minimum > (W[testInd, i] - V[ind,i]):
                    minimum = W[testInd, i] - V[ind,i]
                
                if W[ind, i] == Xh[i]: # maximum point is expanded
                    dim = [91, i]
                else: # minimum point is expanded
                    dim = [92, i]
                    
                
    return dim


def torch_improvedHyperboxContraction(V, W, newCD, testedInd, ind):
    """
    Adjusting min-max points of overlaping regions (9 cases)
    
      V, W = hyperboxContraction(V,W,newCD,testedInd,ind)
  
    INPUT
      V1            Lower bounds of existing hyperboxes
      W1            Upper bounds of existing hyperboxes
      newCD         Special parameters, output from improvedHyperboxOverlapTest
      testedInd     Index of hyperbox to test for overlap with the extended hyperbox
      ind           Index of extended hyperbox	
   
    OUTPUT
      V             Lower bounds of adjusted hyperboxes
      W             Upper bounds of adjusted hyperboxes
    
    """
    
    if newCD[0] == 1 or newCD[0] == 91:
        W[ind, newCD[1]] = (V[testedInd, newCD[1]] + W[ind, newCD[1]]) / 2
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    elif newCD[0] == 2 or newCD[0] == 92:
        V[ind, newCD[1]] = (W[testedInd, newCD[1]] + V[ind, newCD[1]]) / 2
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
    elif newCD[0] == 3 or newCD[0] == 82:
        V[testedInd, newCD[1]] = W[ind, newCD[1]]
    elif newCD[0] == 4 or newCD[0] == 72:
        W[ind, newCD[1]] = V[testedInd, newCD[1]]
    elif newCD[0] == 5 or newCD[0] == 71:
        V[ind, newCD[1]] = W[testedInd, newCD[1]]
    elif newCD[0] == 6 or newCD[0] == 81:
        W[testedInd, newCD[1]] = V[ind, newCD[1]]
        
    return (V, W)

