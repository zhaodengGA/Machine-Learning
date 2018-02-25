#! /usr/bin/python
# _*_ coding: utf-8 _*_
#
# ----------------------
# @Author: Zhao Deng
# ----------------------
#
# File name: Training.py
#
# The purpose of this file is to train the model to obtain the prior probability and conditional probability

import numpy as np
import Testing as test
import copy


def CreateVocabulary(emailWordsin):
    # create a vocabulary list according to the words in emails
    num = len(emailWordsin)
    vocabularySet = set()
    for i in range(0,num):
        vocabularySet = vocabularySet | set(emailWordsin[i])
    vocabulary = list(vocabularySet)
    return vocabulary
    print 'The vocabulary list has been created.'


def training(trainEmailsin, trainCategoryin, vocabularyin, ReEmailCategoryin):
    # calculate the prior probability P(S) = P_S, and the Laplacian Correction smoothing method has been applied,
    # 2 is the number of categories of the email
    num = len(ReEmailCategoryin)
    P_S = float(sum(ReEmailCategoryin) + 1.) / (num + 2.)
    # calculate the conditional probability P(Wi|S) = P_Wi_S, and P(Wi|H) = P_Wi_H.
    # I add 0.02 for all P(Wi|S) in order to avoid all the information being removed by P(Wi|S)=0.
    # Using 0.02 instead of 1 helps to increase the model accuracy, since it reduces the impact of the additional
    # non-exist words.
    WspamCount = np.ones(len(vocabularyin))/50.
    WhamCount = np.ones(len(vocabularyin))/50.
    Wspamtot = 0
    Whamtot = 0
    for i in range(0, len(trainEmailsin)):
        for word in trainEmailsin[i]:
            index = vocabularyin.index(word)
            if trainCategoryin[i] == 1:  # spam email
                WspamCount[index] += 1
                Wspamtot += 1
            if trainCategoryin[i] == 0:  # ham email
                WhamCount[index] += 1
                Whamtot += 1
    # the Laplacian Correction smoothing method has been applied
    P_Wi_S = WspamCount / float(Wspamtot + num)
    P_Wi_H = WhamCount / float(Whamtot + num)
    return P_Wi_S, P_Wi_H, P_S
    print 'Training is Done! P(S), P(Wi|S) and P(Wi|H) have been returned.'

