import pandas as pd
import sklearn as sl
import ipdb
import pickle as pl


def trainOneModel(trainOneModelChoice,X_train,Y_train,targetName,modelUse,modelName,finalModelFolder):
    print '*'*80
    print 'running trainOneModel.py'
    finalModelPath = finalModelFolder+'/final_model.pickle'
    if trainOneModelChoice==0:
        print 'no single final model trained!'
    elif trainOneModelChoice==2:
        print 'load previous trained final one model!'
        with open(finalModelPath,'rb') as handle:
            modelUse = pl.load(handle)
        return modelUse
    elif trainOneModelChoice==1:
        X = X_train
        Y = Y_train
        modelUse.fit(X,Y)
        with open(finalModelPath,'wb') as handle:
            pl.dump(modelUse,handle)
        print 'final one model training done!'
        if modelName in ['gbt','rf','xgb']:
            if modelName == 'xgb':
                featureImportance = modelUse.booster().get_fscore().items()
                featureImportance = pd.DataFrame(featureImportance,columns=['name','importance'])
            else:
                featureImportance = modelUse.feature_importances_
                featureImportance = pd.concat([pd.DataFrame(X.columns,columns=['name']),pd.DataFrame(featureImportance,columns=['importance'])],axis=1)

            featureImportance.sort_values(by='importance',ascending=False,inplace=True)
            featureImportance.index = range(featureImportance.shape[0])
            featureImportance.to_csv('%s/final_single_model_feature_importance.csv'%finalModelFolder,index=False)
            print 'writting down feature importances for final single model done!'
        return modelUse


