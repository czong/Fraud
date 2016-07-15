from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import cross_validation as cv
import seaborn as sb
import pandas as pd
import numpy as np
import ipdb

def test_rf():
    print '*'*80
    calc = False
    if calc == True:
        inputData = pd.read_hdf('./rise_DM_fraud/dev1/preprocessing/preprocessing_result.h5')
        target = 'fpd'
        Y = inputData[target]
        X = inputData.drop(target,axis=1)
        X.fillna(-999,inplace=True)
        clf = RFC(n_estimators = 1000,n_jobs=4,verbose=1)
        clf.fit(X,Y)
        featureImportance_all = pd.concat([pd.DataFrame(X.columns,columns=['name']),pd.DataFrame(clf.feature_importances_,columns=['importance_all'])],axis=1)

        for iter_i in range(1,6):
            skf = cv.StratifiedKFold(y=Y,n_folds=5,shuffle=True,random_state=np.random.randint(0,1000))
            for i,(_,test_index) in enumerate(skf):
                print 'iter %d Fold %d'%(iter_i,i+1)
                test_X  = X.iloc[test_index,:]
                test_Y  = Y[test_index]
                clf = RFC(n_estimators=1000,n_jobs=4,verbose=1)
                clf.fit(test_X,test_Y)
                featureImportance_temp = pd.concat([pd.DataFrame(X.columns,columns=['name']),pd.DataFrame(clf.feature_importances_,columns=['importance_iter_%d_fold_%d'%(iter_i,i+1)])],axis=1)
                featureImportance_all = featureImportance_all.merge(featureImportance_temp,on='name',how='inner')

        featureImportance_all.sort_values(by='importance_all',ascending=False,inplace=True)
        featureImportance_all.to_csv('./rise_DM_fraud/dev1/feature_ranking/feature_importance_rf_all.csv',index=False)
    else:
        featureImportance_all = pd.read_csv('./rise_DM_fraud/dev1/feature_ranking/feature_importance_rf_all.csv')
    # plot the importance
    featureNumShow = 30
    data_plot_1 = featureImportance_all.iloc[:featureNumShow,2:]
    g=sb.boxplot(data=data_plot_1.transpose())
    #g=sb.swarmplot(data=data_plot_1.transpose())
    data_plot_2 = featureImportance_all['importance_all'][:featureNumShow]
    h=plt.plot(range(featureNumShow),data_plot_2,'r')
    plt.savefig('./rise_DM_fraud/dev1/feature_ranking/feature_selection_stability_test_rf.png')

                    
if __name__ == '__main__':
    test_rf()

