# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:18:22 2018

@author: Thanh Tung Khuat

Preprocessing functions helper

"""

import numpy as np
import itertools
from functionhelper.bunchdatatype import Bunch
from functionhelper import UNLABELED_CLASS

dtype = np.float64

def normalize(A, new_range, old_range = None):
    """
    Normalize the input dataset
    
    INPUT
        A           Original dataset (numpy array) [rows are samples, cols are features]
        new_range   The range of data after normalizing
        old_range   The old range of data before normalizing
   
    OUTPUT
        Normalized dataset
    """
    D = A.copy()
    n, m = D.shape
    
    for i in range(m):
        v = D[:, i]
        if old_range is None:
            minv = np.nanmin(v)
            maxv = np.nanmax(v)
        else:
            minv = old_range[0]
            maxv = old_range[1]
        
        if minv == maxv:
            v = np.ones(n) * 0.5;
        else:      
            v = new_range[0] + (new_range[1] - new_range[0]) * (v - minv) / (maxv - minv)
        
        D[:, i] = v;
    
    return D

def replaceMissingValue(Xdata, handling_type = 0):
    """
        Missing value handling by replacing it by another value based on the handling_type parameter
        
            handling_type = 0: Keep nan
                            1: Replace with mean of that feature
                            2: Replace with median of that feature
                            3: Padding zero numbers
            Xdata:      a matrix of input data
            
            return: Xdata was handled
    """
    num_features = Xdata.shape[1]
    
    if handling_type == 1:
        mean_feature = np.nanmean(Xdata, axis=0)
        returned_X = Xdata.copy()
        
        for i in range(num_features):
            returned_X[:, i] = np.where(np.isnan(Xdata[:, i]), mean_feature[i], Xdata[:, i])
                  
    elif handling_type == 2:
        median_feature = np.nanmedian(Xdata, axis=0)
        returned_X = Xdata.copy()
        
        for i in range(num_features):
            returned_X[:, i] = np.where(np.isnan(Xdata[:, i]), median_feature[i], Xdata[:, i])
             
    elif handling_type == 3:
        returned_X = Xdata.copy()
        
        for i in range(num_features):
            returned_X[:, i] = np.where(np.isnan(Xdata[:, i]), 0, Xdata[:, i])
             
    else:
        returned_X = Xdata
    
    return returned_X
    

def loadDataset(path, percentTr, isNorm = False, new_range = [0, 1], old_range = None, class_col = -1):
    """
    Load file containing dataset and convert data in the file to training and testing datasets. Class labels are located in the last column in the file
    Note: Missing value in the input file must be question sign ?
    
        Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(path, percentTr, True, [0, 1])
    
    INPUT
       path             the path to the data file (including file name)
       percentTr        the percentage of data used for training (0 <= percentTr <= 1)
       isNorm           identify whether normalizing datasets or not, True => Normalized
       new_range        new range of datasets after normalization
       old_range        the range of original datasets before normalization (all features use the same range)
       class_col        -1: the class label is the last column in the dataset
                        otherwise: the class label is the first column in the dataset

    OUTPUT
       Xtr              Training dataset
       Xtest            Testing dataset
       patClassIdTr     Training class labels
       patClassIdTest   Testing class labels
       
    """
    
    lstData = []
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' ').replace('?', 'nan'), dtype=dtype, sep=' ').tolist()
            if len(nums) > 0:
                lstData.append(nums)
#            if (a.size == 0):
#                a = nums.reshape(1, -1)
#            else:
#                a = np.concatenate((a, nums.reshape(1, -1)), axis=0)
    A = np.array(lstData, dtype=dtype)
    YA, XA = A.shape
    
    if class_col == -1:
        X_data = A[:, 0:XA-1]
        classId_dat = A[:, -1]
    else:
        # class label is the first column
        X_data = A[:, 1:]
        classId_dat = A[:, 0]
        
    classLabels = np.unique(classId_dat)
    
    # class labels must start from 1, class label = 0 means no label
    if classLabels.size > 1 and np.size(np.nonzero(classId_dat < 1)) > 0:
        classId_dat = classId_dat + 1 + np.min(classId_dat)
        classLabels = classLabels + 1 + np.min(classLabels)

    if isNorm:
        X_data = normalize(X_data, new_range, old_range)
    
    if percentTr != 1 and percentTr != 0:
        noClasses = classLabels.size
        
        Xtr = np.empty((0, XA - 1), dtype=dtype)
        Xtest = np.empty((0, XA - 1), dtype=dtype)

        patClassIdTr = np.array([], dtype=np.int64)
        patClassIdTest = np.array([], dtype=np.int64)
    
        for k in range(noClasses):
            idx = np.nonzero(classId_dat == classLabels[k])[0]
            # randomly shuffle indices of elements belonging to class classLabels[k]
            if percentTr != 1 and percentTr != 0:
                idx = idx[np.random.permutation(len(idx))] 
    
            noTrain = int(len(idx) * percentTr + 0.5)
    
            # Attach data of class k to corresponding datasets
            Xtr_tmp = X_data[idx[0:noTrain], :]
            Xtr = np.concatenate((Xtr, Xtr_tmp), axis=0)
            patClassId_tmp = np.full(noTrain, classLabels[k], dtype=np.int64)
            patClassIdTr = np.append(patClassIdTr, patClassId_tmp)
            
            patClassId_tmp = np.full(len(idx) - noTrain, classLabels[k], dtype=np.int64)
            Xtest = np.concatenate((Xtest, X_data[idx[noTrain:len(idx)], :]), axis=0)
            patClassIdTest = np.concatenate((patClassIdTest, patClassId_tmp))
        
    else:
        if percentTr == 1:
            Xtr = X_data
            patClassIdTr = np.array(classId_dat, dtype=np.int64)
            Xtest = np.array([])
            patClassIdTest = np.array([])
        else:
            Xtr = np.array([])
            patClassIdTr = np.array([])
            Xtest = X_data
            patClassIdTest = np.array(classId_dat, dtype=np.int64)
        
    return (Xtr, Xtest, patClassIdTr, patClassIdTest)

def loadDatasetWithMissingValueHandling(path, percentTr, isNorm = False, new_range = [0, 1], missing_handling_type = 0, class_col = -1):
    """
    Load file containing dataset and convert data in the file to training and testing datasets. Class labels are located in the last column in the file
    Note: Missing value in the input file must be question sign ?
    
        Xtr, Xtest, patClassIdTr, patClassIdTest = loadDataset(path, percentTr, True, [0, 1])
    
    INPUT
       path             the path to the data file (including file name)
       percentTr        the percentage of data used for training (0 <= percentTr <= 1)
       isNorm           identify whether normalizing datasets or not, True => Normalized
       new_range        new range of datasets after normalization
       missing_handling_type        the way of handling missing values:
                                    + 0: Keep nan
                                    + 1: Replace with mean of that feature
                                    + 2: Replace with median of that feature
                                    + 3: Padding zero numbers
       class_col        -1: the class label is the last column in the dataset
                        otherwise: the class label is the first column in the dataset

    OUTPUT
       Xtr              Training dataset
       Xtest            Testing dataset
       patClassIdTr     Training class labels
       patClassIdTest   Testing class labels
       
    """
    
    lstData = []
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' ').replace('?', 'nan'), dtype=dtype, sep=' ').tolist()
            if len(nums) > 0:
                lstData.append(nums)
#            if (a.size == 0):
#                a = nums.reshape(1, -1)
#            else:
#                a = np.concatenate((a, nums.reshape(1, -1)), axis=0)
    A = np.array(lstData, dtype=dtype)
    YA, XA = A.shape
    
    if class_col == -1:
        X_data = A[:, 0:XA-1]
        classId_dat = A[:, -1]
    else:
        # class label is the first column
        X_data = A[:, 1:]
        classId_dat = A[:, 0]
        
    classLabels = np.unique(classId_dat)
    
    # class labels must start from 1, class label = 0 means no label
    if classLabels.size > 1 and np.size(np.nonzero(classId_dat < 1)) > 0:
        classId_dat = classId_dat + 1 + np.min(classId_dat)
        classLabels = classLabels + 1 + np.min(classLabels)

    if isNorm:
        X_data = normalize(X_data, new_range)
        
    X_data = replaceMissingValue(X_data, missing_handling_type)
    
    if percentTr != 1 and percentTr != 0:
        noClasses = classLabels.size
        
        Xtr = np.empty((0, XA - 1), dtype=dtype)
        Xtest = np.empty((0, XA - 1), dtype=dtype)

        patClassIdTr = np.array([], dtype=np.int64)
        patClassIdTest = np.array([], dtype=np.int64)
    
        for k in range(noClasses):
            idx = np.nonzero(classId_dat == classLabels[k])[0]
            # randomly shuffle indices of elements belonging to class classLabels[k]
            if percentTr != 1 and percentTr != 0:
                idx = idx[np.random.permutation(len(idx))] 
    
            noTrain = int(len(idx) * percentTr + 0.5)
    
            # Attach data of class k to corresponding datasets
            Xtr_tmp = X_data[idx[0:noTrain], :]
            Xtr = np.concatenate((Xtr, Xtr_tmp), axis=0)
            patClassId_tmp = np.full(noTrain, classLabels[k], dtype=np.int64)
            patClassIdTr = np.append(patClassIdTr, patClassId_tmp)
            
            patClassId_tmp = np.full(len(idx) - noTrain, classLabels[k], dtype=np.int64)
            Xtest = np.concatenate((Xtest, X_data[idx[noTrain:len(idx)], :]), axis=0)
            patClassIdTest = np.concatenate((patClassIdTest, patClassId_tmp))
        
    else:
        if percentTr == 1:
            Xtr = X_data
            patClassIdTr = np.array(classId_dat, dtype=np.int64)
            Xtest = np.array([])
            patClassIdTest = np.array([])
        else:
            Xtr = np.array([])
            patClassIdTr = np.array([])
            Xtest = X_data
            patClassIdTest = np.array(classId_dat, dtype=np.int64)
        
    return (Xtr, Xtest, patClassIdTr, patClassIdTest)

def loadDatasetWithoutClassLabel(path, percentTr, isNorm = False, new_range = [0, 1]):
    """
    Load file containing dataset without class label and convert data in the file to training and testing datasets.
    
        Xtr, Xtest = loadDatasetWithoutClassLabel(path, percentTr, True, [0, 1])
    
    INPUT
       path             the path to the data file (including file name)
       percentTr        the percentage of data used for training (0 <= percentTr <= 1)
       isNorm           identify whether normalizing datasets or not, True => Normalized
       new_range        new range of datasets after normalization

    OUTPUT
       Xtr              Training dataset
       Xtest            Testing dataset
       
    """
    lstData = []
    with open(path) as f:
        for line in f:
            nums = np.fromstring(line.rstrip('\n').replace(',', ' '), dtype=dtype, sep=' ').tolist()
            if len(nums) > 0:
                lstData.append(nums)
#            if (X_data.size == 0):
#                X_data = nums.reshape(1, -1)
#            else:
#                X_data = np.concatenate((X_data, nums.reshape(1, -1)), axis = 0)
    X_data = np.array(lstData, dtype=dtype)
    if isNorm:
        X_data = normalize(X_data, new_range)
        
    # randomly shuffle indices of elements in the dataset
    numSamples = X_data.shape[0]
    newInds = np.random.permutation(numSamples)
    
    if percentTr != 1 and percentTr != 0:
        noTrain = int(numSamples * percentTr + 0.5)
        Xtr = X_data[newInds[0:noTrain], :]
        Xtest = X_data[newInds[noTrain:], :]
    else:
        if percentTr == 1:
            Xtr = X_data
            Xtest = np.array([])
        else:
            Xtr = np.array([])
            Xtest = X_data
        
    return (Xtr, Xtest)


def saveDataToFile(path, X_data):
    """
    Save data to file
    
    INPUT
        path        The path to the data file (including file name)
        X_data      The data need to be stored
    """
    np.savetxt(path, X_data, fmt='%f', delimiter=', ')   
    

def string_to_boolean(st):
    if st == "True" or st == "true":
        return True
    elif st == "False" or st == "false":
        return False
    else:
        raise ValueError
        

def splitDatasetRndToKPart(Xl, Xu, patClassId, k = 10, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into k parts randomly.
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            k               Number of parts needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            partitionedA    An numpy array contains k sub-arrays, in which each subarray is Bunch datatype:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
    
    numSamples = Xl.shape[0]
    # generate random permutation
    pos = np.random.permutation(numSamples)
    
    # Bin the positions into numClassifier partitions
    anchors = np.round(np.linspace(0, numSamples, k + 1)).astype(np.int64)
    
    partitionedA = np.empty(k, dtype=Bunch)
    
    # divide the training set into k sub-datasets
    for i in range(k):
        partitionedA[i] = Bunch(lower = Xl[pos[anchors[i]:anchors[i + 1]], :], upper = Xu[pos[anchors[i]:anchors[i + 1]], :], label = patClassId[pos[anchors[i]:anchors[i + 1]]])
        
    return partitionedA
    
  
def splitDatasetRndClassBasedToKPart(Xl, Xu, patClassId, k= 10, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into k parts randomly according to each class, where the number of samples of each class is equal among subsets
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            k               Number of parts needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            partitionedA    An numpy array contains k sub-arrays, in which each subarray is Bunch datatype:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
        
    classes = np.unique(patClassId)
    partitionedA = np.empty(k, dtype=Bunch)
    
    for cl in range(classes.size):
        # Find indices of input samples having the same label with classes[cl]
        indClass = patClassId == classes[cl]
        # filter samples having the same class label with classes[cl]
        Xl_cl = Xl[indClass]
        Xu_cl = Xu[indClass]
        pathClass_cl = patClassId[indClass]
        
        numSamples = Xl_cl.shape[0]
        # generate random permutation of positions of selected patterns
        pos = np.random.permutation(numSamples)
        
        # Bin the positions into k partitions
        anchors = np.round(np.linspace(0, numSamples, k + 1)).astype(np.int64)
        
        for i in range(k):
            if cl == 0:
                lower_tmp = Xl_cl[pos[anchors[i]:anchors[i + 1]], :]
                upper_tmp = Xu_cl[pos[anchors[i]:anchors[i + 1]], :]
                label_tmp = pathClass_cl[pos[anchors[i]:anchors[i + 1]]]
                partitionedA[i] = Bunch(lower = lower_tmp, upper = upper_tmp, label = label_tmp)
            else:
                lower_tmp = np.concatenate((partitionedA[i].lower, Xl_cl[pos[anchors[i]:anchors[i + 1]], :]), axis=0)
                upper_tmp = np.concatenate((partitionedA[i].upper, Xu_cl[pos[anchors[i]:anchors[i + 1]], :]), axis=0)
                label_tmp = np.append(partitionedA[i].label, pathClass_cl[pos[anchors[i]:anchors[i + 1]]])
                partitionedA[i] = Bunch(lower = lower_tmp, upper = upper_tmp, label = label_tmp)
        
    return partitionedA


def splitDatasetRndClassBasedTo2Part(Xl, Xu, patClassId, training_rate = 0.5, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into 2 parts randomly according to each class, the proposition training_rate is applied for each class
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            training_rate   The percentage of the number of training samples needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            trainingSet     One object belonging to Bunch datatype contains training data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            validSet        One object belonging to Bunch datatype contains validation data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
        
    classes = np.unique(patClassId)
    trainingSet = None
    validSet = None
    
    for cl in range(classes.size):
        # Find indices of input samples having the same label with classes[cl]
        indClass = patClassId == classes[cl]
        # filter samples having the same class label with classes[cl]
        Xl_cl = Xl[indClass]
        Xu_cl = Xu[indClass]
        pathClass_cl = patClassId[indClass]
        
        numSamples = Xl_cl.shape[0]
        # generate random permutation of positions of selected patterns
        pos = np.random.permutation(numSamples)
        
        # Find the cut-off position
        pivot = int(numSamples * training_rate)
        
        if cl == 0:
            trainingSet = Bunch(lower = Xl_cl[pos[0:pivot]], upper = Xu_cl[pos[0:pivot]], label = pathClass_cl[pos[0:pivot]])
            validSet = Bunch(lower = Xl_cl[pos[pivot:]], upper = Xu_cl[pos[pivot:]], label = pathClass_cl[pos[pivot:]])
        else:
            lower_train = np.concatenate((trainingSet.lower, Xl_cl[pos[0:pivot]]), axis=0)
            upper_train = np.concatenate((trainingSet.upper, Xu_cl[pos[0:pivot]]), axis=0)
            label_train = np.append(trainingSet.label, pathClass_cl[pos[0:pivot]])
            trainingSet = Bunch(lower = lower_train, upper = upper_train, label = label_train)
            
            lower_valid = np.concatenate((validSet.lower, Xl_cl[pos[pivot:]]), axis=0)
            upper_valid = np.concatenate((validSet.upper, Xu_cl[pos[pivot:]]), axis=0)
            label_valid = np.append(validSet.label, pathClass_cl[pos[pivot:]])
            validSet = Bunch(lower = lower_valid, upper = upper_valid, label = label_valid)
            
        
    return (trainingSet, validSet)


def splitDatasetRndTo2Part(Xl, Xu, patClassId, training_rate = 0.5, isNorm = False, norm_range = [0, 1]):
    """
    Split a dataset into 2 parts randomly on whole dataset, the proposition training_rate is applied for whole dataset
    
        INPUT
            Xl              Input data lower bounds (rows = objects, columns = features)
            X_u             Input data upper bounds (rows = objects, columns = features)
            patClassId      Input data class labels (crisp)
            training_rate   The percentage of the number of training samples needs to be split
            isNorm          Do normalization of input training samples or not?
            norm_range      New ranging of input data after normalization, for example: [0, 1]
            
        OUTPUT
            trainingSet     One object belonging to Bunch datatype contains training data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            validSet        One object belonging to Bunch datatype contains validation data with the following attributes:
                                + lower:    lower bounds
                                + upper:    upper bounds
                                + label:    class labels
            
    """
    if isNorm == True:
        Xl = normalize(Xl, norm_range)
        Xu = normalize(Xu, norm_range)
    
    numSamples = Xl.shape[0]
    # generate random permutation
    pos = np.random.permutation(numSamples)
    
    # Find the cut-off position
    pivot = int(numSamples * training_rate)
    
    # divide the training set into 2 sub-datasets
    trainingSet = Bunch(lower = Xl[pos[0:pivot]], upper = Xu[pos[0:pivot]], label = patClassId[pos[0:pivot]])
    validSet = Bunch(lower = Xl[pos[pivot:]], upper = Xu[pos[pivot:]], label = patClassId[pos[pivot:]])
    
    return (trainingSet, validSet)


def read_file_in_chunks_group_by_label(filePath, chunk_index, chunk_size):
    """
    Read data in the file with path filePath in chunks and group data by labels in each chunk
    
        INPUT
            filePath        The path to the file containing data in the hard disk (including file name and its extension)
            chunk_index     The index of chunk needs to read
            chunk_size      The number of data lines in each chunk (except for the last chunk with fewer lines than common maybe) 
            
        OUTPUT
            results         A dictionary contains the needed chunk, where key is label and value is a list of data corresponding to each key
    """
    with open(filePath) as f:
        start = chunk_index * chunk_size
        stop = (chunk_index + 1) * chunk_size
        dic_results = {}
        for line in itertools.islice(f, start, stop):
            if line != None:
                num_data = np.fromstring(line.rstrip('\n').replace(',', ' ').replace('?', 'nan'), dtype=np.float64, sep=' ').tolist()
                lb = num_data[-1]
                if lb in dic_results:
                    dic_results[lb].data.append(num_data[0:-1])
                    dic_results[lb].label.append(lb)
                else:
                    dic_results[lb] = Bunch(data = [num_data[0:-1]], label=[lb])
        
        results = None
        for key in dic_results:
            if results == None:
                results = {}
            results[key] = Bunch(data = np.asarray(dic_results[key].data, dtype=dtype), label = np.asarray(dic_results[key].label, dtype=np.int64))
        
        return results
    
def read_file_in_chunks(filePath, chunk_index, chunk_size):
    """
        Read data in the file with path filePath in chunks and does not group data by label
    
        INPUT
            filePath        The path to the file containing data in the hard disk (including file name and its extension)
            chunk_index     The index of chunk needs to read
            chunk_size      The number of data lines in each chunk (except for the last chunk with fewer lines than common maybe) 
            
        OUTPUT
                            A bunch datatype includes the list of data and labels (properties: data, label)
    """
    with open(filePath) as f:
        start = chunk_index * chunk_size
        stop = (chunk_index + 1) * chunk_size
        returned_res = None
        result = []
        for line in itertools.islice(f, start, stop):
            if line != None and len(line) > 0:
                num_data = np.fromstring(line.rstrip('\n').replace(',', ' ').replace('?', 'nan'), dtype=np.float64, sep=' ').tolist()
                result.append(num_data)
        
        if len(result) > 0:
            input_data = np.array(result, dtype=dtype) # convert data from list to numpy array
            X_data = input_data[:, 0:-1]
            label = input_data[:, -1]        
            returned_res = Bunch(data=X_data, label=label)
            
        return returned_res
    
def convert_missing_value_to_used_format(Xl, Xu, patClass):
    """
        This function is to convert input data with missing values to the formart able to be used in the GFMM networks
                    Xl, Xu, patClass = convert_missing_value_to_used_format(Xl, Xu, patClass)
        INPUT
            Xu          A numpy matrix contains all values (including missing values) for maximum vertices of hyperboxes
            Xl          A numpy matrix contains all values (including missing values) for minimum vertices of hyperboxes
            patClass    A numpy array contains all classess corresponding to input patterns
            
        OUTPUT
            Xu, Xl, patClass are converted to the format able to be used for the functions in GFMM
    """
    Xl = np.where(np.isnan(Xl), 1, Xl)
    Xu = np.where(np.isnan(Xu), 0, Xu)
    patClass = np.where(np.isnan(patClass), UNLABELED_CLASS, patClass)
    
    return (Xl, Xu, patClass)
    