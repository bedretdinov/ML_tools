
import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from __future__ import unicode_literals 
 

class NormalEducationML:
    
    beta = None

    def train(self,X,y):
        X = np.array(X)
        X = np.hstack(  (np.ones((X.shape[0],1)), X) )   
        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) 

    def predict(self,X): 
        X = np.array(X)
        X = np.hstack(  (np.ones((X.shape[0],1)), X) )
        return X.dot(self.beta)
    
    def mse(self,y_pred,y): 
        y_pred = np.array(y_pred)
        y = np.array(y)
        res = np.mean((y - y_pred)**2)
        return res
    
    def rmse(self,y_pred,y):  
        res = np.sqrt(self.mse(y_pred,y)) 
        return res


X = [[1,46],[1,1],[46,3],[3,2]]
y = [1,1,0,0] 


nml = NormalEducationML()
nml.train(X,y)


nml.rmse([1,1,0,0] , nml.predict(X) )