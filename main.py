########################################################
########################################################
########################################################
##### main_v4.py is to try single model with new missing replacement and feature selection methods.
### phase 1: try xgboost with no missing value imputation, in xgboost, in each splitting node, a default direction will be learned for the missing value. This is similar to using surrogate splitting in the normal(unregularized) gradient boosting trees. Note that surrogate splitting is not recommneded in the random forest model: the surrogate variables chosen may not correlated well with the primary variable since only a subset of variables are selected for each splitting (usually small portion of it, like 'sqrt')

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

#####################################################################################################
####################     Configuration     #####################
#####################################################################################################

# general setting
# 0:no action 1:take action 2:load previous result
readData_choice = 0
preprocessing_choice = 2
feature_ranking_choice = 2
hyper_choice = 2
stacking_choice= 0
## train with CV_testing is for small dataset which does not have a separate test dataset
train_with_CV_testing_choice = 1
## train final model is for median and large daaset which does have a separate test dataset
train_final_model_choice = 0
feature_connect_dict_choice = 0
# 0:no action 1:take action 2:load previous result
storage_path = '/esas/SASDataFiles/czong/result'
project = 'rise_DM_fraud'
subProject = 'dev1' 

rawdata_folder = '%s/%s/%s/rawdata' %(storage_path,project,subProject)
preprocess_folder = '%s/%s/%s/preprocessing' %(storage_path,project,subProject)
featureRank_folder = '%s/%s/%s/feature_ranking' %(storage_path,project,subProject)
stacking_folder = '%s/%s/%s/stacking'%(storage_path,project,subProject)
hyper_folder = '%s/%s/%s/hyper' %(storage_path,project,subProject)
final_model_folder = '%s/%s/%s/final_model' %(storage_path,project,subProject)
feature_dict_connection_folder = '%s/%s/%s/feature_dict_connect' %(storage_path,project,subProject)
test_folder = '%s/%s/%s/test' %(storage_path,project,subProject)
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

#### read data module
if readData_choice == 0:
    inputData = []
elif readData_choice ==1:
    # CLR_1
    inputFile = '/esas/SASDataFiles/czong/rise_dm_fraud/riseDMFraudData_CLR_1.csv' 
    inputData = pd.read_csv(inputFile)
    print 'clr_1 column #:%d'%inputData.shape[1]
    # CLR_2
    inputFile = '/esas/SASDataFiles/czong/rise_dm_fraud/riseDMFraudData_CLR_2.csv' 
    tempData = pd.read_csv(inputFile)
    tempData.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','CLR_AppID'],axis=1,inplace=True)
    inputData = pd.merge(left=inputData,right=tempData,left_on='ApplicationNumber',right_on='ApplicationNumber')
    print 'clr_2 column #:%d'%tempData.shape[1]
    # IDA
    inputFile = '/esas/SASDataFiles/czong/rise_dm_fraud/riseDMFraudData_IDA.csv' 
    tempData = pd.read_csv(inputFile)
    tempData.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','IDA_AppID'],axis=1,inplace=True)
    inputData = pd.merge(left=inputData,right=tempData,left_on='ApplicationNumber',right_on='ApplicationNumber')
    print 'ida column #:%d'%tempData.shape[1]
    # RL
    inputFile = '/esas/SASDataFiles/czong/rise_dm_fraud/riseDMFraudData_RL.csv' 
    tempData = pd.read_csv(inputFile)
    tempData.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','AppID'],axis=1,inplace=True)
    inputData = pd.merge(left=inputData,right=tempData,left_on='ApplicationNumber',right_on='ApplicationNumber')
    print 'rl column #:%d'%tempData.shape[1]
    # TG
    inputFile = '/esas/SASDataFiles/czong/rise_dm_fraud/riseDMFraudData_TG.csv' 
    tempData = pd.read_csv(inputFile)
    tempData.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','TG_AppID'],axis=1,inplace=True)
    inputData = pd.merge(left=inputData,right=tempData,left_on='ApplicationNumber',right_on='ApplicationNumber')
    print 'tg column #:%d'%tempData.shape[1]
    # TT
    inputFile = '/esas/SASDataFiles/czong/rise_dm_fraud/riseDMFraudData_TT.csv' 
    tempData = pd.read_csv(inputFile)
    tempData.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','TT_AppID'],axis=1,inplace=True)
    inputData = pd.merge(left=inputData,right=tempData,left_on='ApplicationNumber',right_on='ApplicationNumber')
    print 'tt column #:%d'%tempData.shape[1]
    # TU
    inputFile = '/esas/SASDataFiles/czong/rise_dm_fraud/riseDMFraudData_TU.csv' 
    tempData = pd.read_csv(inputFile)
    tempData.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','TU_AppID'],axis=1,inplace=True)
    inputData = pd.merge(left=inputData,right=tempData,left_on='ApplicationNumber',right_on='ApplicationNumber')
    print 'tu column #:%d'%tempData.shape[1]
    with open(rawdata_folder+'/rawdata.pickle','wb') as handle:
        pickle.dump(inputData,handle)
    print 'saving raw dataset into pickle, finished!'
elif readData_choice == 2:
    with open(rawdata_folder+'/rawdata.pickle','rb') as handle:
        inputData = pickle.load(handle)
    print 'load the existing raw dataset from pickle file, finished!'

##########################################################################################
####################     preprocessing     #####################
##########################################################################################
targetName = 'fpd'
missingRateHighBound = 0.5
categoryUpLimit = 40
fillna = 'None'    # options: 'mean', 'median', '-999', 'None'
var_threshold = 0
scale_enable = False  # when cross validation, no scale in the preprocessing stage
write_en = True

inputAfterPre = pre_processing(preprocessing_choice,inputData,targetName,missingRateHighBound,categoryUpLimit,fillna,var_threshold,scale_enable,write_en,preprocess_folder)


##########################################################################################
####################     feature selection     #####################
##########################################################################################
ranking_method = 'rf'
featureNum = 400

inputAfterSelect = feature_select(feature_ranking_choice,ranking_method,inputAfterPre,targetName,featureRank_folder,featureNum)

##########################################################################################
####################    hyper parameter search     #####################
##########################################################################################
classifierList = ['xgb']  # full list of classifiers: ['xgb','gbt','rf','ERT'], more are coming 
maxIter = 100 

hyperopt_model = hyperopt_search(hyper_choice,inputAfterSelect,targetName,classifierList,maxIter,hyper_folder)

##########################################################################################
#########    train the model with cv testing     #########################################
##########################################################################################
finalSingleModelName = 'xgb'

trainWithCVTest(train_with_CV_testing_choice,inputAfterSelect,targetName,hyperopt_model[finalSingleModelName],finalSingleModelName,final_model_folder)

##########################################################################################
#########    train the final model with the found hyperparameter and all train data     ##
##########################################################################################
finalSingleModelName = 'xgb'

finalSingleModel = trainOneModel(train_final_model_choice,inputAfterSelect,targetName,hyperopt_model[finalSingleModelName],finalSingleModelName,final_model_folder)


##########################################################################################
#########    stacking various models targeting at a more distinguishable model     #######
##########################################################################################
if stacking_choice == 1:
    clfs = {
            'layer1':{'rf':hyperopt_model['rf'],'ERT':hyperopt_model['ERT']},
            'layer2':{'rf':hyperopt_model['rf'],'ERT':hyperopt_model['ERT']}
           }
else:
    clfs = {}
'''
clfs = {
        'layer1':{'gbt':hyperopt_model['gbt'],'rf':hyperopt_model['rf'],'ERT':hyperopt_model['ERT']},
        'layer2':{'gbt':hyperopt_model['gbt'],'rf':hyperopt_model['rf'],'ERT':hyperopt_model['ERT']}
       }
''' 

stackedModel = stackingTrain(stacking_choice,inputAfterPre,targetName,clfs,stacking_folder,cvNum=10)


##########################################################################################
#########  connect the features ranked by importance with the actual meaning from dict  ##
##########################################################################################
dict_fold = '/esas/SASDataFiles/czong/elastic_DM_response'
TU_dict_path = '%s/Elevate Credit Archive - Order #52485 - Final Layout.xlsx'%dict_fold
Exp_dict_path = '%s/EX_AllAttributesDictionary.xlsx'%dict_fold

featureConnectDict(feature_connect_dict_choice,final_model_folder,TU_dict_path,Exp_dict_path,feature_dict_connection_folder)

##########################################################################################
#########    test the model   ############################################################
##########################################################################################
         


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
