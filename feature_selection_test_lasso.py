from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation as cv
import seaborn as sb
import pandas as pd
import numpy as np
import ipdb

def test_lasso():
    alphaNum = 6
    print '*'*80
    inputData = pd.read_hdf('./rise_DM_fraud/dev1/preprocessing/preprocessing_result.h5')
    target = 'fpd'
    Y = inputData[target]
    X = inputData.drop(target,axis=1)
    X.fillna(-999,inplace=True)
    lars_cv = linear_model.LassoLarsCV(cv=6).fit(X,Y)
    skf = cv.StratifiedKFold(y=Y,n_folds=5)
    for i,(_,test_index) in enumerate(skf):
        print 'Fold',i
        test_X  = X.iloc[test_index,:]
        test_Y  = Y[test_index]
        alphas = np.linspace(lars_cv.alphas_[0], .1*lars_cv.alphas_[0],alphaNum)
        clf = linear_model.RandomizedLasso(alphas,random_state=33,n_jobs=1).fit(test_X,test_Y)
        featureImportance = pd.DataFrame(sorted(zip(map(lambda x:round(x,4),clf.scores_),X.columns),reverse=True),columns=['importance','name'])
        featureImportance.to_csv('./rise_DM_fraud/dev1/feature_ranking/feature_importance_lasso_fold_%d.csv'%(i+1),index=False)
                    
if __name__ == '__main__':
    test_lasso()

