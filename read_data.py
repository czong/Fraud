import pandas as pd
import pickle
import ipdb


def read_data(read_choice,read_folder):
    print '*'*80
    print'running read_data.py'
    if read_choice == 0:
        inputData = None
    elif read_choice == 1:
        CLR_1 = pd.read_csv('%s/BK_CLR_1.csv'%read_folder)
        CLR_2 = pd.read_csv('%s/BK_CLR_2.csv'%read_folder)
        IDA = pd.read_csv('%s/BK_IDA.csv'%read_folder)
        RL = pd.read_csv('%s/BK_RL.csv'%read_folder)
        TU = pd.read_csv('%s/BK_TU.csv'%read_folder)
        EXP = pd.read_csv('%s/BK_EXP.csv'%read_folder)

        CLR_1.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','reporting_customer_type','reporting_channel','seriesID','CLR_AppID'],axis=1,inplace=True)
        CLR_2.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','reporting_customer_type','reporting_channel','seriesID','CLR_AppID'],axis=1,inplace=True)
        IDA.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','reporting_customer_type','reporting_channel','seriesID','IDA_AppID'],axis=1,inplace=True)
        RL.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','reporting_customer_type','reporting_channel','seriesID','AppID'],axis=1,inplace=True)
        TU.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','reporting_customer_type','reporting_channel','seriesID','TU_AppID'],axis=1,inplace=True)
        EXP.drop(['Acceptance','pmt','fpd','fraud01','app_decision_dt','dec_date','reporting_customer_type','reporting_channel','seriesID','EX_AppID'],axis=1,inplace=True)
        #CLR_1.drop(['Acceptance','pmt','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','CLR_AppID'],axis=1,inplace=True)
        #CLR_2.drop(['Acceptance','pmt','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','CLR_AppID'],axis=1,inplace=True)
        #IDA.drop(['Acceptance','pmt','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','IDA_AppID'],axis=1,inplace=True)
        #RL.drop(['Acceptance','pmt','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','AppID'],axis=1,inplace=True)
        #TU.drop(['Acceptance','pmt','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','TU_AppID'],axis=1,inplace=True)
        #EXP.drop(['Acceptance','pmt','fraud01','app_decision_dt','dec_date','dec_month','reporting_customer_type','reporting_channel','seriesID','EX_AppID'],axis=1,inplace=True)

        inputData = pd.merge(left=CLR_1,right=CLR_2,how='inner',on='ApplicationNumber')
        inputData = pd.merge(left=inputData,right=TU,how='outer',on='ApplicationNumber')
        inputData = pd.merge(left=inputData,right=IDA,how='outer',on='ApplicationNumber')
        inputData = pd.merge(left=inputData,right=RL,how='outer',on='ApplicationNumber')
        inputData = pd.merge(left=inputData,right=EXP,how='outer',on='ApplicationNumber')


#        target_fpd_df = pd.concat([inputData['fpd'],inputData['fpd_x'],inputData['fpd_y']],axis=1)
        target_fpd_df = pd.concat([inputData['BK_x'],inputData['BK_y']],axis=1)
        target_fpd = target_fpd_df.max(axis=1)
#        inputData.drop(['fpd','fpd_x','fpd_y'],axis=1,inplace=True)
        inputData.drop(['BK_x','BK_y'],axis=1,inplace=True)
        inputData['BK']=target_fpd
        
#        dec_month_df = inputData[['dec_month','dec_month_x','dec_month_y']]                      # time dependence test
        dec_month_df = inputData[['dec_month_x','dec_month_y']]                      # time dependence test
        dec_month = dec_month_df.max(axis=1)                                                     # time dependence test 
#        inputData.drop(['dec_month','dec_month_x','dec_month_y'],axis=1,inplace=True)            # time dependence test
        inputData.drop(['dec_month_x','dec_month_y'],axis=1,inplace=True)            # time dependence test
        inputData['dec_month']=dec_month                                                         # time dependence test

        with open(read_folder+'/rawdata.pickle','wb') as handle:
            pickle.dump(inputData,handle)
        print 'saving raw dataset into pickle, finished!'
    elif read_choice == 2:
        with open(read_folder+'/rawdata.pickle','rb') as handle:
            inputData = pickle.load(handle)
        print 'previous raw data is loaded!'

    return inputData

