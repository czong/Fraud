########################################################
########################################################
########################################################
##### main_v4.py is to try single model with new missing replacement and feature selection methods.
### phase 1: try xgboost with no missing value imputation, in xgboost, in each splitting node, a default direction will be learned for the missing value. This is similar to using surrogate splitting in the normal(unregularized) gradient boosting trees. Note that surrogate splitting is not recommneded in the random forest model: the surrogate variables chosen may not correlated well with the primary variable since only a subset of variables are selected for each splitting (usually small portion of it, like 'sqrt')

from read_data import read_data
from preprocessing import pre_processing
from feature_selection import feature_select
from hyperopt_search import hyperopt_search
from stacking import stackingTrain
from perf_evaluation import evaluate_metrics
from perf_evaluation import get_ks
from perf_evaluation import liftChart
from featureConnectDict import featureConnectDict
from trainOneModel import trainOneModel
from trainWithCVTest import trainWithCVTest
from predict import singleModelPredict,stackModelPredict
from testing import testing
from clustering import clusteringForSegmentationTest
import pandas as pd
import numpy as np
import ipdb
import pdb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import grid_search
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from datetime import datetime
from time import time
import pickle
import os
import re
import seaborn as sb


#####################################################################################################
####################     Configuration     #####################
#####################################################################################################

# general setting
readData_choice = 1  # 0:no action 1:take action 2:load previous result
preprocessing_choice = 1  # 0:no action 1:take action 2:load previous result
feature_ranking_choice = 1
hyper_choice = 1
## train final model is for median and large daaset which does have a separate test dataset
train_final_model_choice = 1
clustering_choice = 1

esas_server = True
if os.uname()[1]=='LXAD16268':
    if esas_server == False:
        storage_path = '/Users/czong/working/Fraud'
    else:
        storage_path = '/Volumes/SAS Data Files/czong/result'
elif re.search('^PRDRSKBDA([0-9]+)',os.uname()[1],re.IGNORECASE):
    storage_path = '/esas/SASDataFiles/czong/result'
project = 'rise_DM_fraud'
subProject = 'dev5_BK'

rawdata_folder = '%s/%s/rawdata' %(storage_path,project)
preprocess_folder = '%s/%s/%s/preprocessing' %(storage_path,project,subProject)
featureRank_folder = '%s/%s/%s/feature_ranking' %(storage_path,project,subProject)
stacking_folder = '%s/%s/%s/stacking'%(storage_path,project,subProject)
hyper_folder = '%s/%s/%s/hyper' %(storage_path,project,subProject)
final_model_folder = '%s/%s/%s/final_model' %(storage_path,project,subProject)
feature_dict_connection_folder = '%s/%s/%s/feature_dict_connect' %(storage_path,project,subProject)
test_folder = '%s/%s/%s/test' %(storage_path,project,subProject)
clustering_folder = '%s/%s/%s/cluster'%(storage_path,project,subProject)
shadow_folder = '%s/%s/%s/shadow' %(storage_path,project,subProject)
if not os.path.exists(storage_path):
    os.makedirs(storage_path)
if not os.path.exists(rawdata_folder):
    os.makedirs(rawdata_folder)
if not os.path.exists(preprocess_folder):
    os.makedirs(preprocess_folder)
if not os.path.exists(featureRank_folder):
    os.makedirs(featureRank_folder)
if not os.path.exists(hyper_folder):
    os.makedirs(hyper_folder)
if not os.path.exists(stacking_folder):
    os.makedirs(stacking_folder)
if not os.path.exists(final_model_folder):
    os.makedirs(final_model_folder)
if not os.path.exists(feature_dict_connection_folder):
    os.makedirs(feature_dict_connection_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
if not os.path.exists(clustering_folder):
    os.makedirs(clustering_folder)
if not os.path.exists(shadow_folder):
    os.makedirs(shadow_folder)

##########################################################################################
####################     read data     #####################
##########################################################################################
inputData = read_data(readData_choice,rawdata_folder)
dec_month = inputData['dec_month']
inputData.drop('dec_month',axis=1,inplace=True)

##########################################################################################
####################     preprocessing     #####################
##########################################################################################
targetName = 'BK'
missingRateHighBound = 0.5
categoryUpLimit = 40
fillna = 'None'    # options: 'mean', 'median', '-999', 'None'
var_threshold = 0
scale_enable = False  # when cross validation, no scale in the preprocessing stage
write_en = True

Y = inputData[targetName]
X = inputData.drop(targetName,axis=1,inplace=False)

#dec_month = X['dec_month']  # time dependence test
#X.drop(['dec_month'],axis=1,inplace=True)  # time dependence test

X,Y = pre_processing(preprocessing_choice,X,Y,targetName,missingRateHighBound,categoryUpLimit,fillna,var_threshold,scale_enable,write_en,preprocess_folder)

##########################################################################################
####################    train, test dataset split     #####################
##########################################################################################
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.3,random_state=33,stratify=Y)
dec_month_train = dec_month[Y_train.index]
dec_month_test = dec_month[Y_test.index]
X_train.index = range(X_train.shape[0])
Y_train.index = range(Y_train.shape[0])
X_test.index = range(X_test.shape[0])
Y_test.index = range(Y_test.shape[0])
dec_month_train.index = range(dec_month_train.shape[0])
dec_month_test.index = range(dec_month_test.shape[0])

#train_index_bool = (dec_month>'2015-12') & (dec_month<'2016-04') # time dependence test
#test_index_bool = dec_month>'2016-03' # time dependence test
##test_index_bool = ~train_index_bool    # time dependence test
#X_train = X.ix[train_index_bool,:]     # time dependence test 
#Y_train = Y[train_index_bool]          # time dependence test
#X_test = X.ix[test_index_bool,:]       # time dependence test
#Y_test = Y[test_index_bool]            # time dependence test


##########################################################################################
####################    feature selection     #####################
##########################################################################################
ranking_method = 'lasso'
featureNum = 400
fill_missing = False
stable_test_rf = False

X_train = feature_select(feature_ranking_choice,ranking_method,X_train,Y_train,targetName,featureRank_folder,featureNum,fill_missing,stable_test_rf)

#XForClustering = X_train.ix[Y_train==1,:]
#clusteringForSegmentationTest(XForClustering)
#ipdb.set_trace()
##########################################################################################
####################    hyper parameter search     #####################
##########################################################################################
classifierList = ['xgb']  # full list of classifiers: ['xgb','gbt','rf','ERT','lasso'], more are coming 
maxIter = 100

hyperopt_model = hyperopt_search(hyper_choice,X_train,Y_train,classifierList,maxIter,hyper_folder)

##########################################################################################
#########    train the final model with the found hyperparameter and all train data     ##
##########################################################################################
finalSingleModelName = 'xgb'

finalSingleModel = trainOneModel(train_final_model_choice,X_train,Y_train,targetName,hyperopt_model[finalSingleModelName],finalSingleModelName,final_model_folder)

fpr_select = False
fpr_target = 0.2
tpr_select = True
tpr_target = 0.5
Y_predict_prob = finalSingleModel.predict_proba(X_train)[:,1]
fpr,tpr,thresholds = metrics.roc_curve(Y_train,Y_predict_prob,1)
if fpr_select == True or tpr_select == True:
    print 'threshold on the probability of fraud is being determined!'
    if tpr_select == False and fpr_select == True:
        thresholdSelect = thresholds[np.argmin(abs(fpr-fpr_target))]
    elif tpr_select == True and fpr_select == False:
        thresholdSelect = thresholds[np.argmin(abs(tpr-tpr_target))]
    elif tpr_select == True and fpr_select == True:
        print 'can\'t set both tpr_select and fpr_select to True'


##########################################################################################
#########    test the model   ############################################################
##########################################################################################
config = [('config:read data choice',readData_choice),('config:preprocess choice',preprocessing_choice),('config:feature ranking choice',feature_ranking_choice),('config:hyper choice',hyper_choice),('config:train final model',train_final_model_choice),('config:os name',os.uname()[1]),('config:project',project),('config:subProject',subProject),('prepro:target',targetName),('prepro:missingRateHighBound',missingRateHighBound),('prepro:categoryUpLimit',categoryUpLimit),('prepro:fillna method',fillna),('prepro:variance thresh',var_threshold),('prepro:scale enable',scale_enable),('prepro:write en',write_en),('f_select:feature num',featureNum),('f_select:fill missing',fill_missing),('f_select:stable test rf',stable_test_rf),('f_select:ranking method',ranking_method),('train:final single model name',finalSingleModelName),('train:threshold',thresholdSelect)]
X_test = X_test[X_train.columns]

test_result,Y_perf = testing(config,ranking_method,finalSingleModelName,finalSingleModel,X_test,Y_test,test_folder,thresholdSelect)
temp1 = pd.concat([dec_month_test,Y_perf],axis=1)
temp2 = pd.concat([dec_month_test,pd.DataFrame(np.ones(dec_month_test.shape[0]))],axis=1)
temp3 = pd.concat([dec_month_test,Y_test],axis=1)
temp4 = pd.concat([dec_month_test,-Y_test+1],axis=1)
count_by_month = temp1.groupby('dec_month').sum()
total_by_month = temp2.groupby('dec_month').sum()
fraud_by_month = temp3.groupby('dec_month').sum()
good_by_month = temp4.groupby('dec_month').sum()
date_month_df = pd.DataFrame(count_by_month.index)
count_by_month.index = range(count_by_month.shape[0])
total_by_month.index = range(total_by_month.shape[0])
fraud_by_month.index = range(fraud_by_month.shape[0])
good_by_month.index = range(good_by_month.shape[0])
percent_by_month = pd.DataFrame(count_by_month.iloc[:,0]/total_by_month.iloc[:,0])
percent_by_month['false_positive'] = pd.DataFrame(count_by_month.iloc[:,1]/good_by_month.iloc[:,0]) 
percent_by_month['false_negative'] = pd.DataFrame(count_by_month.iloc[:,2]/fraud_by_month.iloc[:,0]) 
percent_by_month.columns = ['accuracy','false positive rate','false negative rate']
percent_by_month = pd.concat([date_month_df,percent_by_month],axis=1)
fig = plt.figure()
ax1 = plt.subplot(311)
ax1.set_title('accuracy')
sb.barplot(x='dec_month',y='accuracy',data=percent_by_month,palette='Blues_d')
ax2 = plt.subplot(312)
ax2.set_title('false positive rate')
sb.barplot(x='dec_month',y='false positive rate',data=percent_by_month,palette='Blues_d')
ax3 = plt.subplot(313)
ax3.set_title('false negative rate')
sb.barplot(x='dec_month',y='false negative rate',data=percent_by_month,palette='Blues_d')
fig.tight_layout()
fig.savefig('%s/performance_by_month_fpr_0.2_.png'%test_folder)

ipdb.set_trace()
clusteringForSegmentationTest(clustering_choice,clustering_folder,X_test)

#verification(rawdata_folder,shadow_folder,featureRank_folder,finalModelFolder,thresholdSelect)

if __name__ == '__main__':
    pass
'''
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('input_data','target')
    parser.add_argument('-m','--missing',default=0.5,dest='missingRateHighBound',help='upper bound for missing rate to tolerate')
    parser.add_argument('-c','--category',default=40,dest='categoryUpLimit',help='upper bound for category number to tolerate')
    parser.add_argument('-f','--fillna',default='median',dest='fillna',help='method to fill nan in original dataset')
    parser.add_argument('-v','--var',default=0,dest='var_threshold',help='lower bound for feature\'s variance to tolerate')
    parser.add_argument('-s','--scale',default=False,dest='scale_enable',help='scale features to:mean=0,std=1')
    parser.add_argument('-w','--write',default=True,dest='write_en',help='whether write the processed data into the file')
'''
