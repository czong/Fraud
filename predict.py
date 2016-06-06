import pandas as pd
import sklearn as sl
import pickle 
import ipdb

def singleModelPredict(inputData,targetName,finalModelFolder,targetIndexInPredict):
    finalModelPath = finalModelFolder+'final_model.pickle'
    with open(finalModelPath,'rb') as handle:
        finalSingleModel = pickle.load(handle)
    Y = inputData[targetName]
    X = inputData.drop(targetName,axis=1,inplace=False)
    Y_predict = finalSingleModel.predict(X)
    Y_predict_prob = finalSingleModel.predict_proba(X)
    return Y_predict,Y_predict_prob[:,targetIndexInPredict]
    

def stackModelPredict(inputData,targetName,stackModelsFolder,targetIndexInPredict):
    stackModelsPath = stackModelsFolder+'stackingModel.pickle'
    with open(stackModelsPath,'rb') as handle:
        stackModels = pickle.load(handle)
    Y = inputData[targetName]
    X = inputData.drop(targetName,axis=1,inplace=False)
    print 'start predicting using stack models!'
    layerNum = len(stackModels)
    featureOld = X
    for layer_i in range(layerNum):
        featureNew = []
        for model_i in stackModels['layer%d',layer_i+1]: 
           featureNew.append(list(model_i.predict_prob(featureOld)[targetIndexInPredict])) 
        ipdb.set_trace()
        featureNew = pd.DataFrame(featureNew)
        featureOld = featureNew
    Y_predict_prob = featureNew
    return Y_predict_prob



