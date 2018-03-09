# Naive Bayes Spam Classifier
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
        the point is that the probability for this behavior is distributed equally for each word.

        HERE IS SOMETHING NEW. By changing the '1' in eq.(4) to '0.02', the final result of
        spam classifier error rate decrease 40% from 0.023 to 0.014. The reason is that when we add all
        the words of vocabulary to both spam and ham category, the weight of these non-exist words are
        at the same amplitude level of the real words. In order to lower the extra impact from these
        non-exist words, their weight has been dropped down 50 times. This method will be much more
        effective with less data set. Besides, you can do the saturation test to find the best weight.

    * Since P(Wi|S) normally is a very small number, I take a Logarithmic operation to this formula as
        people usual do in order to avoid the memory underflow error.
        
Divide the data to 10 parts randomly, and each time make one part to the testing set and the remains to 
the training set. The final result is the average of these 10 tests. The average spam classifier error 
rate reaches to 1.4%.
