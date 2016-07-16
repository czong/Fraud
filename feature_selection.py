import pandas as pd
import numpy as np
import ipdb
from sklearn import ensemble
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn import cross_validation as cv
import seaborn as sb

def feature_select(feature_ranking_choice,ranking_method,X_train,Y_train,targetName,featureRank_folder,featureNum,fill_missing,stable_test_rf=False):
    # lasso configuration
    alphaNum = 6
    # random forest configuration
    nTrees = 1000
    njobs = 4
    maxFeaturePercent = 0.1
    nFeaturePlot = 30

    featureRanking_la = featureRank_folder+'/feature_importance_lasso.csv'
    afterSelectData_la = featureRank_folder+'/dataAfterSelect_lasso.h5'
    featureRanking_rf = featureRank_folder+'/feature_importance_rf.csv'
    afterSelectData_rf = featureRank_folder+'/dataAfterSelect_rf.h5'

    print '*'*80
    print 'running feature_selection.py'
    if feature_ranking_choice == 0:
        print 'no feature ranking or selection!'
        return X_train
    elif feature_ranking_choice == 2:
        print 'previous feature ranking and selection result is loaded!'
        if ranking_method == 'lasso':
            featureNames = pd.read_csv(featureRanking_la)['name'][:featureNum]
        elif ranking_method == 'rf':
            featureNames = pd.read_csv(featureRanking_rf)['name'][:featureNum]
        return  X_train[featureNames]
    elif feature_ranking_choice == 1:
        X_train_temp = X_train.copy()
        if X_train_temp.isnull().sum().sum()>0:
            X_train_temp.fillna(-999,inplace=True)
            print 'missing data is temporarily filled by -999 in the feature selection process!'
        #### stability selection: L1-based feature selection 
        if ranking_method == 'lasso':
            ## find best alpha through cross-valiation
            #            lars_cv = linear_model.LassoLarsCV(cv=6).fit(X_train_temp,Y_train)
            ## choose the alpha candidates
            #            alphas = np.linspace(lars_cv.alphas_[0], .1*lars_cv.alphas_[0], alphaNum)
            ## obtain scores of features coming with different alphas and combine them, max() used across all alphas's score
            #            clf1 = linear_model.RandomizedLasso(alpha=alphas, random_state=42,n_jobs=1).fit(X_train_temp,Y_train) 
            clf1 = linear_model.RandomizedLasso(alpha='aic',random_state=33,n_jobs=1,verbose=True).fit(X_train_temp,Y_train) 
            ## sort the scores of features 
            featureImportance = pd.DataFrame(zip(X_train_temp.columns,map(lambda x:round(x,4),clf1.scores_)),columns=['name','importance'])
            featureImportance.sort_values(by='importance',ascending=False,inplace=True)
            featureImportance.index = range(featureImportance.shape[0])
            featureImportance.to_csv(featureRanking_la,index=False)
            if fill_missing == True:
                returnData = pd.concat([Y_train,X_train_temp.ix[:,featureImportance.iloc[:featureNum,0]]],axis=1)
            else:
                returnData = pd.concat([Y_train,X_train.ix[:,featureImportance.iloc[:featureNum,0]]],axis=1)
            print 'Lasso feature ranking finish!'
        elif ranking_method == 'rf':
            if stable_test_rf == True:
                test_rf(X_train_temp,Y_train,nTrees,njobs,maxFeaturePercent,featureRank_folder,nFeaturePlot)
            featureImportanceAndName = get_feature_importance_rf(X_train_temp,Y_train,nTrees,njobs,maxFeaturePercent)
            featureImportanceAndName.sort_values(by='importance',ascending=False,inplace=True)
            featureImportanceAndName.to_csv(featureRanking_rf,index=False) 
            if fill_missing == True: 
                returnData = pd.concat([Y_train,X_train_temp.ix[:,featureImportanceAndName['name'][:featureNum]]],axis=1) 
            else: 
                returnData = pd.concat([Y_train,X_train.ix[:,featureImportanceAndName['name'][:featureNum]]],axis=1)
            print 'RF feature ranking finish!'
        print 'feature ranking done!'
        returnX = returnData.drop(targetName,axis=1,inplace=False)
        return returnX

def test_rf(X,Y,nTrees,njobs,maxFeaturePercent,featureRank_folder,nFeaturePlot):
    print '*'*80
    featureImportance_rf_all = get_feature_importance_rf(X,Y,nTrees,njobs,maxFeaturePercent)
    featureImportance_rf_all.columns = ['name','importance_all']
    for iter_i in range(1,6):
        skf = cv.StratifiedKFold(y=Y,n_folds=5,shuffle=True,random_state=np.random.randint(0,1000))
        for i,(_,test_index) in enumerate(skf):
            print 'iter %d Fold %d'%(iter_i,i+1)
            test_X  = X.iloc[test_index,:]
            test_Y  = Y[test_index]
            featureImportance_rf_temp = get_feature_importance_rf(test_X,test_Y,nTrees,njobs,maxFeaturePercent)
            featureImportance_rf_temp.columns = ['name','importance_iter_%d_fold_%d'%(iter_i,i+1)]
            featureImportance_rf_all = featureImportance_rf_all.merge(featureImportance_rf_temp,on='name',how='inner')

    featureImportance_rf_all.sort_values(by='importance_all',ascending=False,inplace=True)
    featureImportance_rf_all.to_csv('%s/feature_importance_rf_all.csv'%featureRank_folder,index=False)
    # plot the importance
    data_plot_1 = featureImportance_rf_all.iloc[:nFeaturePlot,2:]
    g=sb.boxplot(data=data_plot_1.transpose())
    #g=sb.swarmplot(data=data_plot_1.transpose())
    data_plot_2 = featureImportance_rf_all['importance_all'][:nFeaturePlot]
    h=plt.plot(range(nFeaturePlot),data_plot_2,'r')
    plt.savefig('%s/feature_selection_stability_test_rf.png'%featureRank_folder)



def get_feature_importance_rf(X,Y,nTrees,njobs,maxFeaturePercent):
    rf1 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='gini',max_features=0.1,max_depth=3,n_jobs=njobs,verbose=1,)
    rf2 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='gini',max_features=0.1,max_depth=5,n_jobs=njobs,verbose=1)
    rf3 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='gini',max_features=0.1,max_depth=7,n_jobs=njobs,verbose=1)
    rf4 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='entropy',max_features=0.1,max_depth=3,n_jobs=njobs,verbose=1)
    rf5 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='entropy',max_features=0.1,max_depth=5,n_jobs=njobs,verbose=1)
    rf6 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='entropy',max_features=0.1,max_depth=7,n_jobs=njobs,verbose=1)
    ## train random forest model
    rf1.fit(X,Y)
    rf2.fit(X,Y)
    rf3.fit(X,Y)
    rf4.fit(X,Y)
    rf5.fit(X,Y)
    rf6.fit(X,Y)
    featureStack = np.vstack((rf1.feature_importances_,rf2.feature_importances_,rf3.feature_importances_,rf4.feature_importances_,rf5.feature_importances_,rf6.feature_importances_))
    featureImportance_matrix = pd.DataFrame(featureStack.transpose(),columns=['gini,max_depth=3','gini,max_depth=5','gini,max_depth=7','entropy,max_depth=3','entropy,max_depth=5','entropy,max_depth=7'])
    featureImportance = featureImportance_matrix.mean(axis=1)
    featureNames = pd.DataFrame(X.columns)
    featureImportanceAndName = pd.concat([featureNames,featureImportance],axis=1)
    featureImportanceAndName.columns = ['name','importance']
    return featureImportanceAndName


if __name__=='__main__':
    #where argparse goes, now just placeholder
    pass





    ## lasso stability path
    '''
    alphas_grid, scores_path = linear_model.lasso_stability_path(X_train_temp.as_matrix(), Y_train.as_matrix(), random_state=42, eps=0.05, n_jobs=1,verbose=2)
    plt.figure()
    hg = plt.plot(alphas_grid, scores_path.T)
    plt.xlabel('alpha/max_alpha')
    plt.ylabel('stability score: proportion of times selected')
    plt.title('Stability Scores Path')
    plt.axis('tight')
    plt.show()
    '''
