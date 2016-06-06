import pandas as pd
import numpy as np
import ipdb
from sklearn import ensemble
from sklearn import linear_model

def feature_select(feature_ranking_choice,ranking_method,inputData,targetName,featureRank_folder,featureNum):
    alphaNum = 6
    nTrees = 1000
    featureRanking_la = featureRank_folder+'/feature_importance_lasso.csv'
    afterSelectData_la = featureRank_folder+'/dataAfterSelect_lasso.h5'
    featureRanking_rf = featureRank_folder+'/feature_importance_rf.csv'
    afterSelectData_rf = featureRank_folder+'/dataAfterSelect_rf.h5'

    print '*'*80
    if feature_ranking_choice == 0:
        print 'no feature ranking or selection!'
        return inputData
    elif feature_ranking_choice == 2:
        print 'previous feature ranking and selection result is loaded!'
        if ranking_method == 'lasso':
            return pd.read_hdf(afterSelectData_la,'dataAfterSelect')
        elif ranking_method == 'rf':
            return pd.read_hdf(afterSelectData_rf,'dataAfterSelect')
    elif feature_ranking_choice == 1:
        if inputData.isnull().sum().sum()>0:
            inputData.fillna(-999,inplace=True)
            print 'missing data is temporarily filled by -999 in the feature selection process!'
        Y = inputData[targetName]
        X = inputData.drop([targetName],axis=1,inplace=False)
        #### L1-based feature selection 
        if ranking_method == 'lasso':
            ## find best alpha through cross-valiation
            lars_cv = linear_model.LassoLarsCV(cv=6).fit(X,Y)
            ## choose the alpha candidates
            alphas = np.linspace(lars_cv.alphas_[0], .1*lars_cv.alphas_[0], alphaNum)
            ## obtain scores of features coming with different alphas and combine them, max() used across all alphas's score
            clf1 = linear_model.RandomizedLasso(alpha=alphas, random_state=42,n_jobs=1).fit(X,Y) 
            ## sort the scores of features 
            featureImportance = pd.DataFrame(sorted(zip(map(lambda x:round(x,4),clf1.scores_),X.columns),reverse=True))
            featureImportance.to_csv(featureRanking_la,index=False)
            store = pd.HDFStore(afterSelectData_la)
            store['dataAfterSelect']=pd.concat([Y,X.ix[:featureImportance.iloc[:featureNum,1]]],axis=1)
            print 'Lasso feature ranking finish!'
        elif ranking_method == 'rf':
            rf1 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='gini',max_features=0.1,max_depth=3,n_jobs=4,verbose=1)
            rf2 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='gini',max_features=0.1,max_depth=5,n_jobs=4,verbose=1)
            rf3 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='gini',max_features=0.1,max_depth=7,n_jobs=4,verbose=1)
            rf4 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='entropy',max_features=0.1,max_depth=3,n_jobs=4,verbose=1)
            rf5 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='entropy',max_features=0.1,max_depth=5,n_jobs=4,verbose=1)
            rf6 = ensemble.RandomForestClassifier(n_estimators=nTrees,criterion='entropy',max_features=0.1,max_depth=7,n_jobs=4,verbose=1)
            ## train random forest model
            rf1.fit(X,Y)
            rf2.fit(X,Y)
            rf3.fit(X,Y)
            rf4.fit(X,Y)
            rf5.fit(X,Y)
            rf6.fit(X,Y)
            ## note down the ranking of features based on the importances in different split criteria and max depth
            featureImportanceAverage = (rf1.feature_importances_+rf2.feature_importances_+rf3.feature_importances_+rf4.feature_importances_+rf5.feature_importances_+rf6.feature_importances_)/6
            sortedFeatureImportance = pd.DataFrame(featureImportanceAverage).sort_values(by=0,ascending=False)
            sortedFeatureNames = X.columns[sortedFeatureImportance.index]
            sortedFeatureImportance.index = range(X.shape[1])
            featureImportance = pd.concat([pd.DataFrame(sortedFeatureImportance),pd.DataFrame(sortedFeatureNames)],axis=1)
            featureImportance.to_csv(featureRanking_rf,index=False)
            store = pd.HDFStore(afterSelectData_rf)
            store['dataAfterSelect']=pd.concat([Y,X.ix[:,featureImportance.iloc[:featureNum,1]]],axis=1)
            print 'RF feature ranking finish!'
        return pd.concat([Y,X.ix[:,featureImportance.iloc[:featureNum,1]]],axis=1)
        print 'feature ranking done!'

if __name__=='__main__':
    #where argparse goes, now just placeholder
    pass





    ## lasso stability path
    '''
    alphas_grid, scores_path = linear_model.lasso_stability_path(X.as_matrix(), Y.as_matrix(), random_state=42, eps=0.05, n_jobs=1,verbose=2)
    plt.figure()
    hg = plt.plot(alphas_grid, scores_path.T)
    plt.xlabel('alpha/max_alpha')
    plt.ylabel('stability score: proportion of times selected')
    plt.title('Stability Scores Path')
    plt.axis('tight')
    plt.show()
    '''
