# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:14:44 2019

@author: Thanh Tung Khuat

This is a file to define all utility functions for measuring the performance 
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    """
        AUC ROC Curve Scoring Function for Multi-class Classification
    """
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    
    return roc_auc_score(y_test, y_pred, average=average)
    


