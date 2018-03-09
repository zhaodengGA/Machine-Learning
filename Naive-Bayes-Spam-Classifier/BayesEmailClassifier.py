#! /usr/bin/python
# _*_ coding: utf-8 _*_
#
# ----------------------
# @Author: Zhao Deng
# ----------------------
#
# File name: BayesEmailClassifier.py


'''
            ** Naïve Bayes Spam Classifier **

The probability of an email with words W being a spam could be obtained by the Bayes' Theorem:

        P(S|W) = P(W|S) * P(S) / P(W)   ------ (1)

Where:
    P(S) is a prior probability representing the probability of a spam.
    P(W) is the probability of the words showing in an email together, which could be ignored since
        it is the same for ether spam of ham category.
    P(W|S) is a conditional probability representing the probability of the words showing in a spam.

Note that:
    * As a naïve Bayes classifier, we can assume that each word shown in the email is an independent
        event. Therefore, P(W|S) could be expressed following the joint probability formula.

        P(W|S) = PI[P(Wi|S)]    ------ (2)

        Where the word Wi is an element of the set W, and PI[] means to multiply all variables in the
        square bracket.

    * In order to avoid all the probability information being removed by P(Wi|S)=0, the Laplacian
     Correction smoothing method has been applied, which is described as below:

        P(S) = (D_S + 1) / (D + N)    ------ (3)

        P(Wi|S) = (D_Wi + 1) / (D_S + Ni)    ------ (4)

        which referred to eq.(7.19) and (7.20) of 'Zhou Zhihua, Machine Leaning'. The '1' in the
        denominator helps to avoid the error caused by multiplying zero. This is a reasonable assumption
        since it is like adding all the words of vocabulary to both spam and ham category once in advance,
        the point is that the probability for this behaviour is distributed equally for each word.

        HERE IS SOMETHING NEW I FIND. By changing the '1' in eq.(4) to '0.02', the final result of
        spam classifier error rate decrease 40% from 0.023 to 0.014. The reason is that when we add all
        the words of vocabulary to both spam and ham category, the weight of these non-exist words are
        at the same amplitude level of the real words. In order to lower the extra impact from these
        non-exist words, I change the '1' to '0.02' so as to decrease their weight 50 times. Note that
        this method will be much more effective with less data. Besides, you can do the saturation test
        to find the best weight.

    * Since P(Wi|S) normally is a very small number, I take a Logarithmic operation to this formula as
        people usual do in order to avoid the memory underflow error.

Divide the data to 10 parts randomly, and each time make one part to the testing set and the remains to
the training set. The final result is the average of these 10 testings.

'''


import Testing as test
import Training as train
import numpy as np
import copy
import re


def loadData():
    # load data from Emails.txt
    filename = './emails/Emails.txt'
    f = file(filename)
    while True:
        linewords=[]  # empty the list of the line before reloaded
        line = f.readline()
        # divide the line to two parts by '\t', and delete '\t'
        linetmp = line.strip().split('\t')
        if linetmp[0] == 'spam':   # 1 for spam, 0 for ham
            emailCategory.append(1)
        elif linetmp[0] == 'ham':
            emailCategory.append(0)
        else:
            # exist reading until the end of file
            if len(line) == 0:
                break
        # delete any non-character symbol
        regEx = re.compile(r'[^a-zA-Z]|\d')
        if len(linetmp) > 1:
            linewordstmp = regEx.split(linetmp[1])
        # change all character to lower case, and store the words
        for word in linewordstmp:
            if len(word) > 0:
                linewords.append(word.lower())
        emailWords.append(linewords)


def divideData(i):
    # rearrange the emails randomly
    if i == 0: # the code below only runs once
        print 'There is',len(emailWords), 'emails in total.'
        print '---------------------'
        random_index = int(np.random.uniform(0,len(emailWords)))
        while len(emailWords) > 0:
            ReEmailWords.append(emailWords[random_index])
            ReEmailCategory.append(emailCategory[random_index])
            del(emailWords[random_index])
            del(emailCategory[random_index])
            random_index = int(np.random.uniform(0, len(emailWords)))

    # divide the data to 10 parts, and make one part to testing set and the remains to training set.
    num = len(ReEmailWords) / 10
    ReEmailWordstmp = []
    ReEmailCategorytmp = []
    ReEmailWordstmp = copy.deepcopy(ReEmailWords)
    ReEmailCategorytmp = copy.deepcopy(ReEmailCategory)
    testEmails = copy.deepcopy(ReEmailWordstmp[i*num : (i+1)*num-1])
    testCategory = copy.deepcopy(ReEmailCategorytmp[i*num : (i+1)*num-1])
    del(ReEmailWordstmp[i*num : (i+1)*num-1])
    del(ReEmailCategorytmp[i*num : (i+1)*num-1])
    trainEmails = copy.deepcopy(ReEmailWordstmp)
    trainCategory = copy.deepcopy(ReEmailCategorytmp)
    if i == 1:
        print 'Take the', i, 'st part of the data as testing set.'
    elif i == 2:
        print 'Take the', i, 'nd part of the data as testing set.'
    elif i == 3:
        print 'Take the', i, 'rd part of the data as testing set.'
    else:
        print 'Take the',i,'th part of the data as testing set.'
    return testEmails, testCategory, trainEmails, trainCategory

#------------------------------------------------
# claim global variables
emailWords = []
emailCategory = []
ReEmailWords = []
ReEmailCategory = []


def main():
    loadData()
    vocabulary=train.CreateVocabulary(emailWords)
    errRate_ave = 0.
    for i in range(0,10):
        # divide data into testing and training set
        testEmails, testCategory, trainEmails, trainCategory = divideData(i)
        # train the model to obtain the prior probability and conditional probability
        P_Wi_S, P_Wi_H, P_S = train.training(trainEmails, trainCategory, vocabulary, ReEmailCategory)
        num = len(vocabulary)
        # test the model and get the error rate of the prediction
        errRate = test.testing(testEmails, testCategory, P_Wi_S, P_Wi_H, P_S, vocabulary)
        if i == 1:
            print 'errRate of the', i, 'st test is', errRate
        elif i == 2:
            print 'errRate of the', i, 'nd test is', errRate
        elif i == 3:
            print 'errRate of the', i, 'rd test is', errRate
        else:
            print 'errRate of the', i, 'th test is', errRate
        print '---------------------'
        errRate_ave += errRate
    # take the average of 10 times of testing results as the final error rate
    errRate_ave = errRate_ave / 10.
    print '*********************************************************'
    print 'The error rate of this spam classifier is', errRate_ave
    print ''
    print 'The End!'
    print '*********************************************************'


if __name__ == '__main__':
    main()
