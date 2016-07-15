import argparse
import pandas as pd
import numpy as np
import pdb
import ipdb
import argparse
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import grid_search
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from datetime import datetime
from time import time

###################################### PRE-PROCESSING
#    Control params
# 0. Read the input file
# 1. Manually remove columns which has numerical value but represent ID
# 2. Change date columns to month_before_decision days (remove negative days)
# 3. Extract features from speical columns (bankcard information + ABA bank names)
# 4. Drop columns with too many missing values
# 5. Drop columns with too many category levels
# 6. Dummize the categorical variables 
# 7. Obtain X and Y, and fill NA in X (optional)
# 8. Remove zero variance variables
# 9. Mean and normalize the columns (optional)
# 10. Write the processed result into file (optional)
######################################

def pre_processing(preprocessing_choice,X,Y,targetName,missingRateHighBound,categoryUpLimit,fillna,var_threshold,scale_enable,write_en,preprocess_folder):
    print '*'*80
    print 'running preprocessing.py'
    resultFile = preprocess_folder+'/preprocessing_result.h5'

    if preprocessing_choice == 0:
        print 'no preprocessing is performed!'
        return X,Y
    elif preprocessing_choice == 2:
        print 'previous preprocessed data is loaded!'
        inputData =  pd.read_hdf(resultFile,'dataAfterPre')
        Y = inputData[targetName]
        X = inputData.drop(targetName,axis=1,inplace=False)
        return X,Y
    elif preprocessing_choice == 1:
        print 'start preprocessing!'

        #### 1. Manually remove columns which has numerical value but represent ID.
        columnsCount= X.shape[1]
        columnsNames = ['ApplicationNumber','CBB_score','CF_clearfraudscore','IDA_IDScore','CBB_fisca_dbit_bureau_score','SSN','popularity','IDA_SSN','longevity','IDA_TransID','CBB_bankrout_num_1','zip','age','household_income','gender','marital_status','IDA_IDScoreResultCode1','IDA_IDScoreResultCode2','IDA_IDScoreResultCode3','IDA_IDScoreResultCode4','IDA_IDScoreResultCode5','Portfolio_RISE','IDA_Portfolio_RISE','EX_TransID','EX_SSN','EX_CreateDate','cached','completition_CD','transaction_ID','report_dt','report_time','preamble','ARF_version','reference_nbr','ssn_variation_ind','ssn','premAtt_messageCD']
        columnsNames = [_ for _ in columnsNames if _ in X.columns]
        X.drop(columnsNames,axis=1,inplace=True)
        print '1--%d --> %d : drop ID columns:' % (columnsCount,X.shape[1])
        columnsCount=X.shape[1]


        #### 2. Change date columns to month_before_decision days (remove negative days)
        #### 3. Extract features from speical columns (bankcard information + ABA bank names)

        #### 4. Drop columns with too many missing values
        dropColumns = X.columns[(X.isnull().sum()/float(X.shape[0]))>missingRateHighBound]
        X.drop(dropColumns,axis=1,inplace=True)
        print '4--%d --> %d : drop columns with missing rate > %f' %(columnsCount,X.shape[1],missingRateHighBound)
        columnsCount = X.shape[1]


        #### 5. Drop columns with too many category levels
        categoryColumnNames = X.columns[(X.dtypes==bool) | (X.dtypes==object)]
        dropColumnNames = categoryColumnNames[X[categoryColumnNames].apply(lambda x: x.unique().size)>categoryUpLimit]
        X.drop(dropColumnNames,axis=1,inplace=True)
        print '5--%d --> %d : drop columns with categories\' num > %d'%(columnsCount,X.shape[1],categoryUpLimit)
        columnsCount = X.shape[1]


        #### 6. Dummize the categorical variables 
        ''' two important things to note: 
        1. for categorical variable, there exist '1' and 1, which will cause two dummy variables have the same name
        ###################################### need to confirm with point 2 ###########################
        2. in some column NA shows as '', some show as nan, get_dummies will treat them different, causing you have two dummy variables with the same name'''
        #categoryColumnNames = X.columns[X.dtypes==object]
        #X.ix[:,categoryColumnNames] = X.ix[:,categoryColumnNames].applymap(str) # solve 1
        #X.fillna(np.nan,inplace=True)
        X = pd.get_dummies(X,dummy_na=True)    # dummy_na=False solved 2
        print '6--%d --> %d : get dummy variables' % (columnsCount,X.shape[1])
        columnsCount = X.shape[1]

        #### 7. fill NA in X (optional)
        if fillna == 'mean':
            X.fillna(X.mean(),inplace=True)
        elif fillna == 'median':
            X.fillna(X.median(),inplace=True)
        elif fillna == '-999':
            X.fillna(-999,inplace=True)
        elif fillna == 'None':
            pass
        else:
            raise ValueError('unrecognized fill nan option!')
        print '7--fill nan with %s method' %(fillna,)

      
        #### 8. Remove zero variance variables
        columnIndexBool = X.apply(lambda x: np.nanstd(x)<=var_threshold)
        X = X.drop(X.columns[columnIndexBool],axis=1,inplace=False)
        print '8--%d --> %d : remove zero variance variables' %(columnsCount,X.shape[1])


        #### 9. Mean and normalize the columns (optional,dependent on selected algorithm)
        if scale_enable == True:
            X_col_names = X.columns
            scaler = preprocessing.StandardScaler().fit(X)
            X = pd.DataFrame(scaler.transform(X))
            X.columns = X_col_names
        print '9--scale data to format with mean=0, std=1'

        #### 10. Write the processed result into file
        if write_en == True:
            #pdb.set_trace()
            store = pd.HDFStore(resultFile)
            store['dataAfterPre'] = pd.concat([X,Y],axis=1)
            print '10--finish writing processed data into the file'
            store.close()
        return X,Y
    print '*'*80

if __name__ == '__main__':
    pass
    '''
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('X','target')
    parser.add_argument('-m','--missing',default=0.5,dest='missingRateHighBound',help='upper bound for missing rate to tolerate')
    parser.add_argument('-c','--category',default=40,dest='categoryUpLimit',help='upper bound for category number to tolerate')
    parser.add_argument('-f','--fillna',default='median',dest='fillna',help='method to fill nan in original dataset')
    parser.add_argument('-v','--var',default=0,dest='var_threshold',help='lower bound for feature\'s variance to tolerate')
    parser.add_argument('-s','--scale',default=False,dest='scale_enable',help='scale features to:mean=0,std=1')
    parser.add_argument('-w','--write',default=True,dest='write_en',help='whether write the processed data into the file')
    '''



