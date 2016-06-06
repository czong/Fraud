import pandas as pd
import numpy as np
import ipdb
import pickle
from sklearn.cross_validation import StratifiedKFold
from perf_evaluation import evaluate_metrics
from matplotlib import pyplot as plt
import numpy as np

def trainWithCVTest(train_with_CV_testing_choice,inputAfterSelect,targetName,modelUse,finalSingleModelName,final_model_folder):
    cvNum = 10
    if train_with_CV_testing_choice == 0:
        pass
    elif train_with_CV_testing_choice == 1:
        Y = inputAfterSelect[targetName]
        X = inputAfterSelect.drop(targetName,axis=1)
        load = True
        if load == False:
            test_index_bool_df =[]
            for iter_i in range(10):
                skf = StratifiedKFold(Y,cvNum)
                for i, (trainIndex,testIndex) in enumerate(skf):
                    modelUseTemp = modelUse
                    print 'iter_%d,fold%d'%(iter_i+1,i+1)
                    X_train = X.iloc[trainIndex,:]
                    Y_train = Y[trainIndex]
                    X_test  = X.iloc[testIndex,:]
                    Y_test  = Y[testIndex]
                    modelUseTemp.fit(X_train,Y_train)
                    Y_predict = modelUseTemp.predict(X_test)
                    Y_predict_prob = modelUseTemp.predict_proba(X_test)[:,1]
                    perf = evaluate_metrics(Y_test,Y_predict,Y_predict_prob,'cv_fold_%d_test'%(i+1),final_model_folder,True,True)
                    #print 'fold_%d performance:'%(i+1)
                    #print pd.DataFrame([perf])
                    if perf['auc']<0.7:
                        print 'auc < 0.7 in %d,%d' %(iter_i+1,i+1)
                        test_index_bool = pd.DataFrame(skf.test_folds==i)
                        if test_index_bool_df.__class__ == list:
                            test_index_bool_df = test_index_bool
                        else:
                            test_index_bool_df = pd.concat([test_index_bool_df,test_index_bool],axis=1)
            with open('%s/test_bool_10_cv.pickle'%final_model_folder,'wb') as handle:
                pickle.dump(test_index_bool_df,handle)
        elif load == True:
            with open('%s/test_bool_10_cv.pickle'%final_model_folder,'rb') as handle:
                test_index_bool_df = pickle.load(handle)
        testIndex = test_index_bool_df.sum(axis=1)==10
        trainIndex = test_index_bool_df.sum(axis=1)!=10
        X_train = X.ix[trainIndex,:]
        Y_train = Y[trainIndex]
        X_test  = X.ix[testIndex,:]
        Y_test  = Y[testIndex]
        modelUse.fit(X_train,Y_train)
        Y_predict = modelUse.predict(X_test)
        Y_predict_prob = modelUse.predict_proba(X_test)[:,1]
        fraud_predict_prob = Y_predict_prob[Y_test.as_matrix()==1]
        nonfraud_predict_prob = Y_predict_prob[Y_test.as_matrix()==0]

        binsNum = 10
        hist_fraud, bins_fraud = np.histogram(fraud_predict_prob,bins=binsNum)
        hist_non,   bins_non   = np.histogram(nonfraud_predict_prob,bins=binsNum)
        cdf_fraud = np.cumsum(hist_fraud)/sum(hist_fraud)
        cdf_nonfraud = np.cumsum(hist_non)/sum(hist_non)
        plt.figure(1)
        plt.plot(bins_fraud[:binsNum],cdf_fraud,'r',bins_non[:binsNum],cdf_nonfraud,'b')
        plt.show()

        ipdb.set_trace()





