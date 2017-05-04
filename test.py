#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:09:47 2017

@author: yaopan
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import svm

import sklearn
from sklearn import decomposition
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def plot_1_by_10_images(images, figsize=None):
    plt.figure(figsize=(10,4))
    for i in range(10):
        ax = plt.subplot(1,10, i+1)
        plt.imshow(images[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    
total = 1000
decoded_imgs = []
for i in range(total):
    decoded_imgs.append(sae.predict(X_test[i].reshape(1,-1)))
    
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# reconstruct error
error = np.zeros(total)
for i in range(total-1):
    error[i] = (np.sum((X_test[i]-decoded_imgs[i])**2)/784)
sorted_error = np.sort(error)
small_thres = sorted_error[10]
large_thres = sorted_error[total-10]

normal_idx = np.where(error <= small_thres)
abnormal_idx = np.where(error >= large_thres)

# show top 10 normal
plot_1_by_10_images(X_test[normal_idx[0][0:10]])

# show top 10 abnormal
plot_1_by_10_images(X_test[abnormal_idx[0][0:10]])

def plot_top_normal_abnormal(digit):
# reconstruct error for digit i
    err_digit = []
    err_idx = []
    for i in range(total-1):
        if labels_test[i] == digit:
            err_digit.append(error[i])
            err_idx.append(i)
            
    sorted_error = np.sort(err_digit)
    small_thres = sorted_error[10]
    large_thres = sorted_error[len(err_digit)-10]
    
    normal_idx = np.where(err_digit <= small_thres)
    abnormal_idx = np.where(err_digit >= large_thres)
    good = []
    for i in range(10):
        good.append(X_test[err_idx[normal_idx[0][i]]])
    plot_1_by_10_images(good)
    
    bad = []
    for i in range(10):
        bad.append(X_test[err_idx[abnormal_idx[0][i]]])
    plot_1_by_10_images(bad)

plot_top_normal_abnormal(6)
        
correct = 0
for i in range(total):
    softmax = sae.encoder.predict(X_test[i].reshape(1,-1))[0]
    digit = np.where(softmax == np.max(softmax))[0][0]
    if digit == labels_test[i]:
        correct = correct + 1
print(correct/total)
  