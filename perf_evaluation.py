import pandas as pd
import numpy as np
import ipdb
from sklearn import metrics
import matplotlib.pyplot as plt

############ show the performance metrics of one classifier over 10-fold cross valiation
def get_ks(Y_true,score):
    score = pd.Series(score)
    score.sort(ascending=False,inplace=True)
    Y_true = pd.Series(Y_true)
    Y_true.index = range(len(Y_true))
    chunck_size = np.ceil(score.shape[0]/10.)
    temp_target_pdf =[]
    temp_nontarget_pdf = []
    temp_target_cdf =[]
    temp_nontarget_cdf = []
    ks=[]
    for _ in range(10):
        if _<9:
            start_index,end_index = _*chunck_size,(_+1)*chunck_size
        else:
            start_index,end_index = _*chunck_size,score.shape[0]
        temp_index = score.index[start_index:end_index]
        temp_Y_true = Y_true[temp_index]
        temp_true_rate=(temp_Y_true==1).sum()/float((Y_true==1).sum())
        temp_target_pdf.append(temp_true_rate)
        temp_negt_rate = (temp_Y_true==0).sum()/float((Y_true==0).sum())
        temp_nontarget_pdf.append(temp_negt_rate)
    temp_target_cdf = np.cumsum(temp_target_pdf)
    temp_nontarget_cdf = np.cumsum(temp_nontarget_pdf)
    ks = [temp_target_cdf[temp]-temp_nontarget_cdf[temp] for temp in range(10)]   
    #pdb.set_trace()
    return np.max(ks)

def liftChart(Y_true,score,nBins=10):
    liftChartTable = {}
    score = pd.Series(score)
    Y_true = pd.Series(Y_true)
    score.sort(ascending=False,inplace=True)
    Y_true.index = range(len(Y_true))
    chunck_size = int(np.ceil(score.shape[0]/float(nBins)))
    mailer = []
    min_score = []
    max_score = []
    target_pdf =[]
    nontarget_pdf = []
    target_cdf =[]
    nontarget_cdf = []
    for _ in range(nBins):
        if _<nBins-1:
            start_index,end_index = _*chunck_size,(_+1)*chunck_size
        else:
            start_index,end_index = _*chunck_size,score.shape[0]
        mailer.append(end_index-start_index)
        min_score.append(min(score[start_index:end_index])*1000)
        max_score.append(max(score[start_index:end_index])*1000)
        temp_index = score.index[start_index:end_index]
        temp_Y_true = Y_true[temp_index]
        temp_true_rate=(temp_Y_true==1).sum()/float((Y_true==1).sum())
        target_pdf.append(temp_true_rate)
        temp_negt_rate = (temp_Y_true==0).sum()/float((Y_true==0).sum())
        nontarget_pdf.append(temp_negt_rate)
    target_cdf = np.cumsum(target_pdf)
    nontarget_cdf = np.cumsum(nontarget_pdf)
    ks = [target_cdf[temp]-nontarget_cdf[temp] for temp in range(nBins)]   

    #pdb.set_trace()

    liftChartTable['#Mailer'] = mailer
    liftChartTable['%Mailer'] = (pd.DataFrame(mailer)/float(Y_true.shape[0])).iloc[:,0].tolist()
    liftChartTable['cumul. %Mailer'] = (pd.DataFrame(np.cumsum(mailer))/float(Y_true.shape[0])).iloc[:,0].tolist()
    liftChartTable['Min Score'] = min_score
    liftChartTable['Max Score'] = max_score
    liftChartTable['Actual Marg RespRate'] = (pd.DataFrame(target_pdf)*(Y_true==1).sum()/pd.DataFrame(mailer)).iloc[:,0].tolist()
    liftChartTable['Cumul. RespRate'] = (pd.DataFrame(target_cdf)*(Y_true==1).sum()/pd.DataFrame(np.cumsum(mailer))).iloc[:,0].tolist()
    liftChartTable['Cumul. % Resp'] = target_cdf
    liftChartTable['Lift'] = (pd.DataFrame(target_cdf)/pd.DataFrame(liftChartTable['cumul. %Mailer'])-1).iloc[:,0].tolist()
    liftChartTable['ks'] = ks
    liftChartTable_df = pd.DataFrame(liftChartTable)
    return liftChartTable_df

def evaluate_metrics(Y_true,Y_predict,Y_predict_prob,ranking_method,modelName,test_title,test_folder,prob_en=False,confusion_en=False):
    # '1' represents positive here
    perfTemp = {}
    perfTemp['accu']=metrics.accuracy_score(Y_true,Y_predict)
    perfTemp['recall']=metrics.recall_score(Y_true,Y_predict)
    perfTemp['precision']=metrics.precision_score(Y_true,Y_predict)
    perfTemp['f1']=metrics.f1_score(Y_true,Y_predict)
    confusionMatrix = metrics.confusion_matrix(Y_true,Y_predict)
    perfTemp['TP']=float(confusionMatrix[1][1])/(confusionMatrix[1][1]+confusionMatrix[1][0])
    perfTemp['TN']=float(confusionMatrix[0][0])/(confusionMatrix[0][0]+confusionMatrix[0][1])
    perfTemp['FP']=1-perfTemp['TN']
    perfTemp['FN']=1-perfTemp['TP']
    if prob_en == True:
        perfTemp['ks']=get_ks(Y_true,Y_predict_prob)
        perfTemp['auc']=metrics.roc_auc_score(Y_true,Y_predict_prob)
        perfTemp['log_loss']=metrics.log_loss(Y_true,Y_predict_prob)
        fpr,tpr,thresholds  = metrics.roc_curve(Y_true,Y_predict_prob,1)
        fig = plt.figure()
        plt.plot(fpr,tpr,label='ROC curve (area = %0.3f)' %perfTemp['auc'])
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(test_title)
        plt.legend(loc='lower right')
        fig.savefig('%s/%s_%s_%s.png'%(test_folder,ranking_method,modelName,test_title))
    if confusion_en == True:
        perfTemp['confusion'] = confusionMatrix
    #pdb.set_trace()
    return perfTemp 

if __name__ == '__main__':
    pass

         

