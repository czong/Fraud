import os
import ipdb
import pandas as pd
import numpy as np
from perf_evaluation import evaluate_metrics
from sklearn import metrics

def testing(config,ranking_method,modelName,model,X_test,Y_test,test_folder,threshold):
    print '*'*80
    print 'running testing.py'
    Y_predict_prob = model.predict_proba(X_test)[:,1]     # 1 is fraud
    Y_predict = pd.DataFrame([1 if _>=threshold else 0 for _ in Y_predict_prob])
    Y_false_positive = pd.DataFrame([1 if (Y_predict.iloc[_,0]==1) & (Y_test[_]==0) else 0 for _ in range(Y_predict.shape[0])])
    Y_false_negative = pd.DataFrame([1 if (Y_predict.iloc[_,0]==0) & (Y_test[_]==1) else 0 for _ in range(Y_predict.shape[0])])
    Y_correct = pd.DataFrame([1 if Y_predict.iloc[_,0]==Y_test[_] else 0 for _ in range(Y_predict.shape[0])])
    Y_perf = pd.concat([Y_correct,Y_false_positive,Y_false_negative],axis=1)
    Y_perf.columns = ['correct','false_positive','false_negative']
    performance = evaluate_metrics(Y_test,Y_predict,Y_predict_prob,ranking_method,modelName,'roc_curve',test_folder,True,False)
    performance_df = pd.DataFrame.from_dict(performance,'index').reset_index()
    performance_df.columns = [0,1]
    config_df = pd.DataFrame(config)
    result = pd.concat([config_df,performance_df],axis=0)
    result.columns=['configuration','value']
    result.index = range(result.shape[0])
    if not os.path.exists('%s/performance.csv'%test_folder):
        result.to_csv('%s/performance.csv'%test_folder)
    else:
        result_all = pd.read_csv('%s/performance.csv'%test_folder)
        result_all = pd.concat([result_all,result],axis=1)
        result_all.to_csv('%s/performance.csv'%test_folder)
    print result
    return result, Y_perf
