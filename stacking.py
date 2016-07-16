import pandas as pd
import numpy as np
import ipdb
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import grid_search
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


def stackingTrain(stacking_choice,inputDF,targetName,clfs,stacking_folder,cvNum=10):
    ''' clfs is a dict whose keys are 'layer%d'. Each layer itself is another dict whose keys are model names, e.g., 'rf', and the value is the model with specified hyperparameter. '''
    print '*'*80
    print 'running stacking.py'
    stacking_path = stacking_folder+'/stackingModel.pickle'
    if stacking_choice==0:
        print 'no stacking!'
    elif stacking_choice==2:
        print 'load previous trained stacking models!'
        with open(stacking_path,'rb') as handle:
            stackedModel = pickle.load(handle)
        return stackedModel 
    elif stacking_choice==1:
        print 'start train stacking models!'
        layerNum = len(clfs)
        # input coerced to DataFrame
        if inputDF.__class__ != pd.core.frame.DataFrame:
            inputDF = pd.DataFrame(inputDF)
            print 'the input dataset is coerced to dataframe, however column labels are not specificed'

        # split the held-in and held-out dataset
        trainData,testData = train_test_split(inputDF,test_size=0.3)
        trainData.index = range(trainData.shape[0])
        testData.index = range(testData.shape[0])
        Y_hold_in = trainData[targetName]
        X_hold_in = trainData.drop(targetName,axis=1)
        Y_hold_out = testData[targetName]
        X_hold_out = testData.drop(targetName,axis=1)

        # define the output layed models
        modelTrained = {}
        skf = list(StratifiedKFold(Y_hold_in,cvNum))
    
        for layer_i in range(len(clfs)):
            modelTrained.update({'layer%d'%(layer_i+1):[]})
            if layer_i>0:
                X_hold_in = pd.DataFrame(Y_test_predict_matrix)
                X_hold_out = pd.DataFrame(Y_hold_out_predict_matrix)

            models = clfs['layer%d'%(layer_i+1)].values()
            modelNames = clfs['layer%d'%(layer_i+1)].keys()

            Y_test_predict_matrix = np.zeros((trainData.shape[0],len(models)))
            Y_hold_out_predict_matrix = np.zeros((testData.shape[0],len(models)))

            for j, clf in enumerate(models):
                print j, clf
                Y_hold_out_predict_inFold = np.zeros((X_hold_out.shape[0],len(skf)))
                temp_clf_list = []
                for i,(trainIndex,testIndex) in enumerate(skf):
                    print 'Fold', i
                    X_train_inFold = X_hold_in.iloc[trainIndex,:]
                    Y_train_inFold = Y_hold_in[trainIndex]
                    X_test_inFold = X_hold_in.iloc[testIndex,:]
                    Y_test_inFold = Y_hold_in[testIndex]
                    clf.fit(X_train_inFold,Y_train_inFold)
                    temp_clf_list.append(clf)
                    Y_test_predict_inFold = clf.predict_proba(X_test_inFold)[:,1]
                    Y_test_predict_matrix[testIndex,j] = Y_test_predict_inFold
                    Y_hold_out_predict_inFold[:,i] = clf.predict_proba(X_hold_out)[:,1]
                modelTrained['layer%d'%(layer_i+1)].extend(temp_clf_list)
                Y_hold_out_predict_matrix[:,j] = Y_hold_out_predict_inFold.mean(1)
                auc_j = roc_auc_score(Y_hold_out,((Y_hold_out_predict_matrix[:,j]-Y_hold_out_predict_matrix[:,j].min())/(Y_hold_out_predict_matrix[:,j].max()-Y_hold_out_predict_matrix[:,j].min())))
                print 'layer %d: auc of %s: %.3f' %(layer_i,modelNames[j],auc_j)
            print 'layer %d finish!'%layer_i

        #### final layer to blend all models
        print 'Stacking'
        modelTrained.update({'layer%d'%(len(clfs)+1):[]})
        clf = LogisticRegression()
        clf.fit(Y_test_predict_matrix,Y_hold_in)
        Y_hold_out_final_predict = clf.predict_proba(Y_hold_out_predict_matrix)[:,1]
        modelTrained['layer%d'%(len(clfs)+1)].append(clf)
        with open(stacking_path,'wb') as handle:
            pickle.dump(modelTrained,handle)
        print 'stacking model training done!'

        print "Linear stretch of predictions to [0,1]"
        Y_hold_out_final_predict = (Y_hold_out_final_predict-Y_hold_out_final_predict.min())/(Y_hold_out_final_predict.max() - Y_hold_out_final_predict.min())
        auc = roc_auc_score(Y_hold_out,Y_hold_out_final_predict)
        print 
        print 'AUC after stacking is %.3f'%auc
        
        return modelTrained 

