import pandas as pd
import numpy as np
import ipdb
import pickle
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import grid_search
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from datetime import datetime
from time import time
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials
import xgboost as xgb
import warnings


def hyperopt_search(hyper_choice,X_train,Y_train,classifierList,maxIter,hyper_folder):
    print '*'*80
    print 'running hyperopt_search'
    warnings.filterwarnings('ignore')
    parameters = {
                  'rf':{
                     'criterion':['gini','entropy'],
                     'min_samples_leaf':[6,8],
                     'max_features':[0.5,0.2,0.1,'sqrt'],
                     'max_depth':range(3,11)
                     },
                  'gbt':{
                     ## 'learning_rate' and 'subsample' are continous distribution not defined here 
                     'max_depth':range(3,11),
                     'n_estimators':np.arange(50,300,10),
                     'min_samples_leaf':[4,6,8] 
                     }, 
                  'ERT':{
                     'criterion':['gini','entropy'],
                     'min_samples_leaf':[4,6,8],
                     'max_features':[0.5,0.2,0.1,'sqrt'],
                     'max_depth':[4,6,8]           
                     },
                  'xgb':{
                     'max_depth':range(3,9),
                     'n_estimators':np.arange(50,300,10),
                     'min_child_weight':range(1,30)
                     },
                  'lasso':{
                     }
                 }
    Y = Y_train
    X = X_train
    hyper_detail_path = hyper_folder+'/hyperDetail.pickle'
    model_path_xgb = hyper_folder+'/model_xgb.pickle'
    model_path_gbt = hyper_folder+'/model_gbt.pickle'
    model_path_rf = hyper_folder+'/model_rf.pickle'
    model_path_ERT = hyper_folder+'/model_ERT.pickle'
    model_path_lasso = hyper_folder+'/model_lasso.pickle'
    hyperopt_model = {}
    hyperDetailDict = {}
    n_jobs = 5
    n_cv = 10
    verbose = 1
    n_estimators_rf = 1000
    n_estimators_ERT = 1000
    if hyper_choice == 2:
        if 'xgb' in classifierList:
            with open(model_path_xgb,'rb') as handle:
                hyperopt_model['xgb']=pickle.load(handle)
        if 'gbt' in classifierList:
            with open(model_path_gbt,'rb') as handle:
                hyperopt_model['gbt']=pickle.load(handle)
        if 'rf' in classifierList:
            with open(model_path_rf,'rb') as handle:
                hyperopt_model['rf']=pickle.load(handle)
        if 'ERT' in classifierList:
            with open(model_path_ERT,'rb') as handle:
                hyperopt_model['ERT']=pickle.load(handle)
        if 'lasso' in classifierList:
            with open(model_path_lasso,'rb') as handle:
                hyperopt_model['lasso']=pickle.load(handle)
        print 'previous searched hyper-tuned models is loaded!'
        return hyperopt_model
    elif hyper_choice == 1:
        if 'lasso' in classifierList:
            space4lasso = {
                'alpha':hp.uniform('alpha',0,1)
            }
            def func_lasso(params):
                clf = Lasso(normalize=True)
                auc = cross_validation.cross_val_score(clf,X.as_matrix(),Y.as_matrix(),scoring='roc_auc',cv=n_cv,n_jobs=1)
                return {'loss':-auc.mean(),'status':STATUS_OK}
            trials = Trials()
            print 'start hyperopt for lasso'
            startTime = time()
            best = fmin(fn=func_lasso,space=space4lasso,algo=tpe.suggest,max_evals=maxIter,trials=trials)
            optimalModel = Lasso(normalize=True,alpha=best['alpha'])
            print 'finish hyperopt for lasso'
            print 'time used is %d seconds' %(time()-startTime)
            print 'best parameters:'
            hyperDetailDict['lasso']={}
            print best
            print optimalModel
            print 'best auc:'
            aucTemp = max([-temp['result']['loss'] for temp in trials.trials])
            print aucTemp 
            hyperDetailDict['lasso']['best_param']=best
            hyperDetailDict['lasso']['best_model']=optimalModel
            hyperDetailDict['lasso']['auc']=aucTemp
            hyperDetailDict['lasso']['trials']=trials.trials
            hyperopt_model['lasso']=optimalModel
            with open(model_path_lasso,'wb') as handle:
                pickle.dump(optimalModel,handle)

        if 'xgb' in classifierList:
            space4xgb = {
                'max_depth':hp.choice('max_depth',parameters['xgb']['max_depth']),
                'n_estimators':hp.choice('n_estimators',parameters['xgb']['n_estimators']),
                'min_child_weight':hp.choice('min_child_weight',parameters['xgb']['min_child_weight']),
                'learning_rate':hp.uniform('learning_rate',0.01,0.1),
                'subsample':hp.uniform('subsample',0.01,1),
                'colsample_bytree':hp.uniform('colsample_bytree',0.01,1),
                'reg_alpha':hp.uniform('reg_alpha',0,3),
                'reg_lambda':hp.uniform('reg_lambda',0,3)
            }
            def func_xgb(params):
                clf = xgb.XGBClassifier(silent=True,nthread=n_jobs,seed=333,objective='binary:logistic',**params)
                auc = cross_validation.cross_val_score(clf,X.as_matrix(),Y.as_matrix(),scoring='roc_auc',cv=n_cv,n_jobs=n_jobs)
                return {'loss':-auc.mean(),'status':STATUS_OK}
            trials = Trials()
            print 'start hyperopt for xgboost'
            startTime = time()
            best = fmin(fn=func_xgb,space=space4xgb,algo=tpe.suggest,max_evals=maxIter,trials=trials)
            optimalModel = xgb.XGBClassifier(silent=True,nthread=n_jobs,seed=333,objective='binary:logistic',learning_rate=best['learning_rate'],subsample=best['subsample'],max_depth=parameters['xgb']['max_depth'][best['max_depth']],n_estimators=parameters['xgb']['n_estimators'][best['n_estimators']],colsample_bytree=best['colsample_bytree'],min_child_weight=parameters['xgb']['min_child_weight'][best['min_child_weight']],reg_alpha=best['reg_alpha'],reg_lambda=best['reg_lambda'])
            print 'finish hyperopt for xgboost'
            print 'time used is %d seconds' %(time()-startTime)
            print 'best parameters:'
            hyperDetailDict['xgb']={}
            print best
            print optimalModel
            print 'best auc:'
            aucTemp = max([-temp['result']['loss'] for temp in trials.trials])
            print aucTemp 
            hyperDetailDict['xgb']['best_param']=best
            hyperDetailDict['xgb']['best_model']=optimalModel
            hyperDetailDict['xgb']['auc']=aucTemp
            hyperDetailDict['xgb']['trials']=trials.trials
            hyperopt_model['xgb']=optimalModel
            with open(model_path_xgb,'wb') as handle:
                pickle.dump(optimalModel,handle)

        if 'rf' in classifierList:
            space4rf = {
                'criterion':hp.choice('criterion',parameters['rf']['criterion']),
                'min_samples_leaf':hp.choice('min_samples_leaf',parameters['rf']['min_samples_leaf']),
                'max_features':hp.choice('max_features',parameters['rf']['max_features']),
                'max_depth':hp.choice('max_depth',parameters['rf']['max_depth'])
                }
            def func_rf(params):
                clf = ensemble.RandomForestClassifier(n_jobs=n_jobs,verbose=verbose,n_estimators=n_estimators_rf,**params)
                auc = cross_validation.cross_val_score(clf,X,Y,scoring='roc_auc',cv=n_cv,n_jobs=n_jobs)
                return {'loss':-auc.mean(),'status':STATUS_OK}
            trials = Trials()
            print 'start hyperopt for random forest'
            startTime = time()
            best = fmin(fn=func_rf,space=space4rf,algo=tpe.suggest,max_evals=maxIter,trials=trials)
            optimalModel = ensemble.RandomForestClassifier(n_jobs=n_jobs,verbose=verbose,n_estimators=n_estimators_rf,criterion=parameters['rf']['criterion'][best['criterion']],min_samples_leaf=parameters['rf']['min_samples_leaf'][best['min_samples_leaf']],max_features=parameters['rf']['max_features'][best['max_features']],max_depth=parameters['rf']['max_depth'][best['max_depth']])
            print 'finish hyperopt for random forest'
            print 'time used is %d seconds' %(time()-startTime)
            print 'best parameter:'
            hyperDetailDict['rf']={}
            print best
            print optimalModel
            print 'best auc:'
            aucTemp = max([-temp['result']['loss'] for temp in trials.trials])
            print aucTemp 
            hyperDetailDict['rf']['best_param']=best
            hyperDetailDict['rf']['best_model']=optimalModel
            hyperDetailDict['rf']['auc']=aucTemp
            hyperDetailDict['rf']['trials']=trials.trials
            hyperopt_model['rf']=optimalModel
            with open(model_path_rf,'wb') as handle:
                pickle.dump(optimalModel,handle)

        if 'gbt' in classifierList:
            space4gbt = {
                'learning_rate':hp.uniform('learning_rate',0.01,0.1),
                'subsample':hp.uniform('subsample',0.1,1.0),
                'max_depth':hp.choice('max_depth',parameters['gbt']['max_depth']),
                'n_estimators':hp.choice('n_estimators',parameters['gbt']['n_estimators']),
                'min_samples_leaf':hp.choice('min_samples_leaf',parameters['gbt']['min_samples_leaf'])
                }
            def func_gbt(params):
                clf = ensemble.GradientBoostingClassifier(verbose=verbose,max_features='sqrt',**params)
                auc = cross_validation.cross_val_score(clf,X,Y,scoring='roc_auc',cv=n_cv,n_jobs=n_jobs)
                return {'loss':-auc.mean(),'status':STATUS_OK}
            trials = Trials()
            print 'start hyperopt for gradient boosting'
            startTime = time()
            best = fmin(fn=func_gbt,space=space4gbt,algo=tpe.suggest,max_evals=maxIter,trials=trials)
            optimalModel = ensemble.GradientBoostingClassifier(verbose=verbose,max_features='sqrt',learning_rate=best['learning_rate'],subsample=best['subsample'],max_depth=parameters['gbt']['max_depth'][best['max_depth']],n_estimators=parameters['gbt']['n_estimators'][best['n_estimators']],min_samples_leaf=parameters['gbt']['min_samples_leaf'][best['min_samples_leaf']])
            print 'finish hyperopt for gradient boosting'
            print 'time used is %d seconds' %(time()-startTime)
            print 'best parameters:'
            hyperDetailDict['gbt']={}
            print best
            print optimalModel
            print 'best auc:'
            aucTemp = max([-temp['result']['loss'] for temp in trials.trials])
            print aucTemp 
            hyperDetailDict['gbt']['best_param']=best
            hyperDetailDict['gbt']['best_model']=optimalModel
            hyperDetailDict['gbt']['auc']=aucTemp
            hyperDetailDict['gbt']['trials']=trials.trials
            hyperopt_model['gbt']=optimalModel
            with open(model_path_gbt,'wb') as handle:
                pickle.dump(optimalModel,handle)

        if 'ERT' in classifierList:
            space4ERT = {
                'criterion':hp.choice('criterion',parameters['ERT']['criterion']),
                'min_samples_leaf':hp.choice('min_samples_leaf',parameters['ERT']['min_samples_leaf']),
                'max_features':hp.choice('max_features',parameters['ERT']['max_features']),
                'max_depth':hp.choice('max_depth',parameters['ERT']['max_depth'])
                }
            def func_ERT(params):
                clf = ensemble.ExtraTreesClassifier(n_jobs=n_jobs,verbose=verbose,n_estimators=n_estimators_ERT,**params)
                auc = cross_validation.cross_val_score(clf,X,Y,scoring='roc_auc',cv=n_cv,n_jobs=n_jobs)
                return {'loss':-auc.mean(),'status':STATUS_OK}
            trials = Trials()
            print 'start hyperopt for extremely randomized trees'
            startTime = time()
            best = fmin(fn=func_ERT,space=space4ERT,algo=tpe.suggest,max_evals=maxIter,trials=trials)
            optimalModel = ensemble.ExtraTreesClassifier(n_jobs=n_jobs,verbose=verbose,n_estimators=n_estimators_ERT,criterion=parameters['ERT']['criterion'][best['criterion']],min_samples_leaf=parameters['ERT']['min_samples_leaf'][best['min_samples_leaf']],max_features=parameters['ERT']['max_features'][best['max_features']],max_depth=parameters['ERT']['max_depth'][best['max_depth']])
            
            print 'finish hyperopt for extremely randomized trees'
            print 'time used is %d seconds' %(time()-startTime)
            print 'best parameters:'
            hyperDetailDict['ERT']={}
            print best
            print optimalModel
            print 'best auc:'
            aucTemp = max([-temp['result']['loss'] for temp in trials.trials])
            print aucTemp 
            hyperDetailDict['ERT']['best_param']=best
            hyperDetailDict['ERT']['best_model']=optimalModel
            hyperDetailDict['ERT']['auc']=aucTemp
            hyperDetailDict['ERT']['trials']=trials.trials
            hyperopt_model['ERT']=optimalModel
            with open(model_path_ERT,'wb') as handle:
                pickle.dump(optimalModel,handle)
        with open(hyper_detail_path,'wb') as handle:
            pickle.dump(hyperDetailDict,handle)
        print 'hyper parameter search done!'
    return hyperopt_model

if __name__=='__main__':
    pass

    
    '''
    parameters_SVM_linear = {'C':[1e-3,1e-2,1e-1,1,1e1,1e2,1e3],'kernel':['linear']}
    parameters_SVM_rbf = {'C':[1e-3,1e-2,1e-1,1,1e1,1e2,1e3],'kernel':['rbf'],'gamma':[1e-3,1e-2,1e-1,1,1e1,1e2,1e3]}
    parameters_SVM_poly = {'C':[1e-3,1e-2,1e-1,1,1e1,1e2,1e3],'kernel':['poly'],'degree':[2,3,4,5],'gamma':[1e-3,1e-2,1e-1,1,1e1,1e2,1e3    ]}
    parameters_SVM_sigmoid = {'C':[1e-3,1e-2,1e-1,1,1e1,1e2,1e3],'kernel':['sigmoid'],'gamma':[1e-3,1e-2,1e-1,1,1e1,1e2,1e3]}
    parameters_lasso = {'alpha':[0.005,0.01,0.05],'fit_intercept':[True,False]}
    '''
