import ipdb
import pandas as pd
import numpy as np

def featureConnectDict(feature_connect_dict_choice,final_model_folder,TU_dict_path,Exp_dict_path,feature_dict_connection_folder):
    feature_importance_path = final_model_folder+'/final_single_model_feature_importance.csv'
    feature_dict_connection_path = feature_dict_connection_folder+'/featureDictConnect.csv'
    if feature_connect_dict_choice==0:
        print 'no feature meaning linking!'
    elif feature_connect_dict_choice==1:
        ''' this script simply match the physical meaning of the variables with the variables' importance in the NL model. The merge tag is the variable name. Two inputs, one is TU dictionary and Experian dictionary, the other one is the variables the model choose in an descending order of importance.'''
        TU_dict = pd.read_excel(TU_dict_path)
        Exp_dict = pd.read_excel(Exp_dict_path)
        f_importance = pd.read_csv(feature_importance_path)

        ipdb.set_trace()
        tu_hit = [1 if item in TU_dict['Output Name:'].tolist() else 0 for item in f_importance['name']]
        exp_hit = [1 if item in Exp_dict['NAME'].tolist() else 0 for item in f_importance['name']] 

        df_output = pd.concat([f_importance,pd.DataFrame(tu_hit,columns=['tu_hit']),pd.DataFrame(exp_hit,columns=['exp_hit'])],axis=1)    

        if ((df_output['tu_hit']==0) & (df_output['exp_hit']==0)).sum()>0:
            print 'there are variabls can\'t be found:'
            var_missing = df_output.ix[(df_output['tu_hit']==0) & (df_output['exp_hit']==0),:]
            print var_missing
        if ((df_output['tu_hit']==1) & (df_output['exp_hit']==1)).sum()>0:
            print 'there are variables shown in both TU and Exp:'
            var_dupli = df_output.ix[(df_output['tu_hit']==1) & (df_output['exp_hit']==1),:]
            print var_dupli

        desp_opt = pd.DataFrame(np.zeros(df_output.shape[0]),columns=['Buerau'])
        desp = pd.DataFrame(np.zeros(df_output.shape[0]),columns=['description'])
        for _ in list(df_output.index):
            if ((df_output['exp_hit'][_]==0) & (df_output['tu_hit'][_]==0)): 
                desp_opt.iloc[_,0]='missing'
                desp.iloc[_,0]='NA'
            else:
                if ((df_output['exp_hit'][_]==1) & (df_output['tu_hit'][_]==1)):
                    desp_opt.iloc[_,0]='duplicate'
                    desp.iloc[_,0]='NA'
                elif df_output['exp_hit'][_]==1:
                    desp_opt.iloc[_,0]='exp'
                    #set_trace()
                    desp.iloc[_,0]=(Exp_dict['DESCRIPTION'][Exp_dict['NAME']==f_importance['name'][_]]).tolist()[0]
                else:
                    desp_opt.iloc[_,0]='tu'
                    desp.iloc[_,0]=(TU_dict['Description:'][TU_dict['Output Name:']==f_importance['name'][_]]).tolist()[0]

        df_output = pd.concat([df_output,desp_opt,desp],axis=1)
        df_output.to_csv(feature_dict_connection_path)
        print 'feature meaning linking done!'

if __name__ == '__main__':
    # where argparse could go, now just a placeholder
    pass
