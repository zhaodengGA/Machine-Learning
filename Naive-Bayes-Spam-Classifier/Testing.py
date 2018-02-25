#! /usr/bin/python
# _*_ coding: utf-8 _*_
#
# ----------------------
# @Author: Zhao Deng
# ----------------------
#
# File name: Testing.py
#
# The purpose of this file is to test the model and get the error rate of the prediction

import Training as train
import numpy as np


def testing(testEmailsin, testCategoryin, P_Wi_Sin, P_Wi_Hin, P_Sin, vocabularyin):
    num = len(testEmailsin)
    errCount = 0.
    for i in range(0, num):
        WordCount = np.zeros(len(vocabularyin))
        P_S_W = 0.
        P_H_W = 0.
        for word in testEmailsin[i]:
            index = vocabularyin.index(word)
            WordCount[index] += 1
        # due to the assumption of independence, the joint probability is expressed as multiplying the probability of each event
        # take a Logarithmic operation to the multiplying since P(Wi|S) is a very small number
        P_S_W = sum(np.log(P_Wi_Sin[:])*WordCount[:]) + np.log(P_Sin)
        P_H_W = sum(np.log(P_Wi_Hin[:])*WordCount[:]) + np.log(1-P_Sin)
        ## The formulas above have the same accuracy of the formulas below but with faster speed.
        #for n in range(0, len(vocabularyin)):
        #    if WordCount[n] != 0:
        #        P_S_W = P_S_W + (np.log(P_Wi_Sin[n])+np.log(WordCount[n]))
        #        P_H_W = P_H_W+ (np.log(P_Wi_Hin[n])+np.log(WordCount[n]))
        #P_S_W += np.log(P_Sin)
        #P_H_W += np.log(1 - P_Sin)
        if P_S_W > P_H_W:
            whetherSpam = True
        else:
            whetherSpam = False

        if whetherSpam == testCategoryin[i]:
            errCount += 1
    errRate = 1. - errCount / float(num)
    return errRate

