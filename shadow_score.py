import ipdb
import pandas as pd
import numpy as np
from 


def verification(rawdata_folder,shadow_folder,featureRank_folder,finalModel_folder,ranking_method,featureNum):
    verify_data = pd.read_sas('%s/riseDM_shadow_CLR_IDA_RL_TU.sas7bdat'%rawdata_folder)
    verify_dec_date = verify_data['dec_date']
    verify_data.drop(['Acceptance','pmt','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','CLR_AppID'],axis=1,inplace=True)
    featureNames = pd.read_csv('%s/feature_importance_%s.csv'%(featureRank_folder,ranking_method))['name'][:featureNum]
    inputData = verify_data[featureNames]

    
