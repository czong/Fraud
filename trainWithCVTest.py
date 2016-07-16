import pandas as pd
import numpy as np
import ipdb
import pickle
from sklearn.cross_validation import StratifiedKFold
from perf_evaluation import evaluate_metrics
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

def trainWithCVTest(train_with_CV_testing_choice,X_train_origin,Y_train_origin,modelUse,finalSingleModelName,final_model_folder):
    cvNum = 10
    load = False
    if train_with_CV_testing_choice == 0:
        pass
    elif train_with_CV_testing_choice == 1:
        X = X_train_origin
        Y = Y_train_origin
        if load == False:
            test_index_bool_df =[]
            for iter_i in range(10):
                #rand_num = np.random.randint(1,100)
                skf = StratifiedKFold(Y,cvNum,shuffle=True)
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
        Y_predict_prob = modelUse.predict_proba(X_test)[:,0]
        fraud_predict_prob = Y_predict_prob[Y_test.as_matrix()==1]
        nonfraud_predict_prob = Y_predict_prob[Y_test.as_matrix()==0]

        ipdb.set_trace()
        # save the CDF on nonfraud probability on fraud and nonfraud data in the problem dataset
        plt.subplot(211)
        plt.hist(x=fraud_predict_prob,bins=30,range=(0,1),normed=True,cumulative=True,color='r')
        plt.title('CDF on nonfraud probability')
        plt.legend('fraud')
        plt.subplot(212)
        plt.hist(x=nonfraud_predict_prob,bins=30,range=(0,1),normed=True,cumulative=True,color='b')
        plt.legend('nonfraud')
        plt.savefig('%s/CDF_of_nonfraud_probability_for_fraud_and_nonfraud_on_narrowed_down_data.png'%final_model_folder)

        ipdb.set_trace()        
        X_test_fraud = X_test.ix[Y_test==1,:]
        X_test_nonfraud = X_test.ix[Y_test==0,:]
        missing_rate_fraud = X_test_fraud.isnull().sum(axis=1)/X_test_fraud.shape[0]
        missing_rate_nonfraud = X_test_nonfraud.isnull().sum(axis=0)/X_test_nonfraud.shape[0]
        plt.subplot(211)
        sb.distplot(a=missing_rate_fraud)
        plt.title('missing_rate_histogram')
        plt.legend('fraud')
        plt.subplot(212)
        sb.distplot(a=missing_rate_nonfraud)
        plt.legend('nonfraud')
        plt.savefig('%s/missing_rate_hist_for_fraud_and_nonfraud_on_narrowed_down_data.png'%final_model_folder)









