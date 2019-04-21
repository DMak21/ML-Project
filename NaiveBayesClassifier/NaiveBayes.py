import numpy as np
import pandas as pd

class NaiveBayes:
    
    def __init__(self, Type='Gaussian', prior=None):
        '''
        To initialize the Type and prior attributes
        '''
        self.Type = Type
        self.prior = prior
        if self.Type not in ['Gaussian','Multinomial']:
            raise KeyError('Unexpected value for \'type\': \'type\' must be either of \'Gaussian\' or \'Multinomial\'.')
        if self.Type != pd.core.frame.DataFrame
            raise TypeError('\'prior\' must be a pandas DataFrame (pandas.core.frame.DataFrame).')
    
    def splitDataset(self, splitRatio=0.7):
        '''
        To split the dataset into a train and a test set
        '''
        trainSet = prior[:int(len(prior) * splitRatio)]
        testSet = prior[int(len(prior) * splitRatio):]
        return trainSet, testSet

    def gaussianprob(x):
        '''
        To find the normal probability
        '''
        y = np.pow((x - np.mean(x,axis=0))/np.std(x,axis=0),2)
        return np.exp(y/2)/np.sqrt(2*np.pi*(std**2))
    
    def multinomialprob(x):
        '''
        To find the multinomial probability
        '''
        s = np.sum(x,axis=0)
        return x/s
    
    def fit(self, dat):
        '''
        To fit the model to the priors
        '''
        probabilities = {}
        if (self.Type=='Gaussian'):
            calculateProbability = gaussianprob
        if(self.Type=='Multinomial'):
            calculateProbability = multinomialprob
        s = self.prior.groupby(self.prior.columns.values[-1]).describe().iloc[:-1,1:3]
        s = self.prior.drop(columns=[self.prior.columns.values[-1]],axis=0)
        for i in range(len(s)):
                classValue = s.index.values[i]
                probabilities[classValue] = 1
                for i in range(len(s)):
                    mean = s.iloc[i,:].values[0]
                    stdev = s.iloc[i,:].values[1]
                    x = data.iloc[i]
                    probabilities[classValue] *= calculateProbability(x)
        return probabilities

    def predict(self, data):
        '''
        To find the posteriors
        '''
        s = data.groupby(data.columns.values[-1]).describe().iloc[:-1,1:3]
        s = data.drop(columns=[data.columns.values[-1]],axis=0)
        probabilities = fit(self)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities:
            if (bestLabel is None) or (probability > bestProb):
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def getPredictions(self, testSet):
        '''
        To return the posteriors in a Pandas DataFrame
        '''
        predictions = pd.DataFrame()
        for i in range(len(testSet)):
            result = predict(self, testSet[i])
            predictions.append(result)
        return predictions

    def getAccuracy(self, testSet, predictions):
        '''
        To see the accuracy of the predictions
        '''
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0