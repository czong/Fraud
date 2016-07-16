import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF

def fillNaWithRFProximityTrain(DFWithNan,targetName,initNumFillMethod='median',maxIter=6):
    # input type coerce
    if DFWithNan.__class__ != pd.core.frame.DataFrame:
        DFWithNan = pd.DataFrame(DFWithNan)
        print 'input is coerced to dataframe, however column labels are not specified!'            
    
    # features and target ready for the model
    Y = DFWithNan[targetName]
    X = DFWithNan.drop(targetName,axis=1)

    # initial nan fill
    if initNumFillMethod=='median':
        DFWithNan.fillna(DFWithNan.median(),inplace=True)
    elif initNumFillMethod=='mean':
        DFWithNan.fillna(DFWithNan.mean(),inplace=True)
    elif initNumFillMethod.__class__==int or initNumFillMethod.__class__==float:
        DFWithNan.fillna(initNumFillMethod,inplace=True)
    else:
        print 'please check initial continuous variable fill method, it\'s neither \'median\', \'mean\' or any int/float number'
        
    # build random forest model to build proximity matrix
    rf = RF(n_estimators=100,verbose=1,n_jobs=4,oob_score=True)
    rf.fitf =    

