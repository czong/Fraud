import os
import sys
import ipdb
import time
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist

def clusteringForSegmentationTest(clustering_choice,clustering_folder,X):
    print '*'*80
    print 'running clustering.py'
    if clustering_choice == 2:
        with open('%s/cluster_process.pickle'%clustering_folder) as h:
            Z = pickle.load(h)
            plot_dendrogram(Z,clustering_folder,False)
    elif clustering_choice == 1:
        if X.isnull().sum().sum()>0:
            X.fillna(X.median(),inplace=True)
            print 'in clustering, nan is found and filled'
        XForClustering = pd.DataFrame(preprocessing.normalize(X,norm='l2',axis=0))
        XForClustering.columns = X.columns
        print 'start clustering'
        Z = linkage(XForClustering,method='ward',metric='euclidean')
        with open('%s/cluster_process.pickle'%clustering_folder,'wr') as h:
            pickle.dump(Z,h)
        ipdb.set_trace()
        plot_dendrogram(Z)

######## hierarchical clustering
def plot_dendrogram(Z,clustering_folder,plot_enable=True):
    fig = plt.figure(figsize=(25,10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True, show_contracted=True, leaf_rotation=90., leaf_font_size=8.,)
    fig.savefig('%s/hierarchical_clustering_dendrogram.png'%clustering_folder)
    if plot_enable == True:
        plt.show()

def plot_dimension_sep_metric(sep_metric):
    plt.figure(figsize=(25,10))
    plt.title('clusters separation metric for each dimension')
    plt.xlabel('dimension index')
    plt.ylabel('separation metric')
    plt.plot(range(len(sep_metric)),sep_metric)
    plt.show()

def dimension_separation_perf_evaluate(cluster1_samples,cluster2_samples):
    eps = 1e-8
    sep_metric = []
    for col_i in range(XForClustering.shape[1]):
        temp1 = cluster1_samples.iloc[:,col_i]
        temp2 = cluster2_samples.iloc[:,col_i]
        #if np.std(temp1)==0 or np.std(temp2)==0:
        #    ipdb.set_trace()
        sep_metric_temp = abs(np.mean(temp1)-np.mean(temp2))/(np.std(temp1)+np.std(temp2)+eps)
        sep_metric.append(sep_metric_temp)
    return sep_metric

'''
###### based on the dendrogram, we perform 2-cluster analysis and 4-cluster analysis
### 2-cluster analysis
k=3
clusters = fcluster(Z,k,criterion='maxclust')
cluster_size_count = [clusters.tolist().count(temp) for temp in set(clusters)]
cluster1_samples = XForClustering.ix[clusters==2,:]
cluster2_samples = XForClustering.ix[clusters==3,:]
sep_metric = dimension_separation_perf_evaluate(cluster1_samples,cluster2_samples)
#plot_dimension_sep_metric(sep_metric)
temp = pd.concat([pd.DataFrame(sep_metric),pd.DataFrame(XForClustering.columns)],axis=1)
temp.columns = ['sep_metric','feature']
temp_sorted = temp.sort_values(by='sep_metric',ascending=False)
temp_sorted.index = range(0,temp_sorted.shape[0])
temp_sorted.to_csv('2_cluster_feature_ranking.csv')

XDummy_cluster1_samples = XDummy.ix[clusters==2,:]
XDummy_cluster2_samples = XDummy.ix[clusters==3,:]
featureNSelect=20
separation_median = [(np.nanmedian(XDummy_cluster1_samples.ix[:,colName]),np.nanmedian(XDummy_cluster2_samples.ix[:,colName])) for colName in temp_sorted.ix[:featureNSelect-1,'feature'].tolist()]
separation_std = [(np.nanstd(XDummy_cluster1_samples.ix[:,colName]),np.nanstd(XDummy_cluster2_samples.ix[:,colName])) for colName in temp_sorted.ix[:featureNSelect-1,'feature'].tolist()]
separation_median_df = pd.DataFrame(separation_median,columns=['median_cluster_1','median_cluster_2'])
separation_std_df = pd.DataFrame(separation_std,columns=['std_cluster_1','std_cluster_2'])
(pd.concat([temp_sorted.ix[:featureNSelect-1,'feature'],separation_median_df,separation_std_df],axis=1)).to_csv('2_cluster_top_variable_median_std.csv')

### 4-cluster analysis
k= 9
sep_metric_list = []
clusters = fcluster(Z,k,criterion='maxclust')
cluster_size_count = [clusters.tolist().count(temp) for temp in set(clusters)]
cluster1_samples = XForClustering.ix[clusters==2,:]
cluster2_samples = XForClustering.ix[clusters==3,:]
cluster3_samples = XForClustering.ix[clusters==6,:]
cluster4_samples = XForClustering.ix[clusters==8,:]
clusterList = [cluster1_samples,cluster2_samples,cluster3_samples,cluster4_samples]
for i in range(len(clusterList)):
    for j in range(i+1,len(clusterList)):
        sep_metric_temp = dimension_separation_perf_evaluate(clusterList[i],clusterList[j])
        sep_metric_list.append(sep_metric_temp)
        print 'i:%d,j:%d' %(i,j)
sep_metric = pd.DataFrame(sep_metric_list).mean()
#plot_dimension_sep_metric(sep_metric)
temp = pd.concat([pd.DataFrame(sep_metric),pd.DataFrame(XForClustering.columns)],axis=1)
temp.columns = ['sep_metric','feature']
temp_sorted = temp.sort_values(by='sep_metric',ascending=False)
temp_sorted.index = range(0,temp_sorted.shape[0])
temp_sorted.to_csv('4_cluster_feature_ranking.csv')

XDummy_cluster1_samples = XDummy.ix[clusters==2,:]
XDummy_cluster2_samples = XDummy.ix[clusters==3,:]
XDummy_cluster3_samples = XDummy.ix[clusters==6,:]
XDummy_cluster4_samples = XDummy.ix[clusters==8,:]
separation_mean = [(np.nanmean(XDummy_cluster1_samples.ix[:,colName]),np.nanmean(XDummy_cluster2_samples.ix[:,colName]),np.nanmean(XDummy_cluster3_samples.ix[:,colName]),np.nanmean(XDummy_cluster4_samples.ix[:,colName])) for colName in temp_sorted.ix[:featureNSelect-1,'feature'].tolist()]
separation_median = [(np.nanmedian(XDummy_cluster1_samples.ix[:,colName]),np.nanmedian(XDummy_cluster2_samples.ix[:,colName]),np.nanmedian(XDummy_cluster3_samples.ix[:,colName]),np.nanmedian(XDummy_cluster4_samples.ix[:,colName])) for colName in temp_sorted.ix[:featureNSelect-1,'feature'].tolist()]
separation_std = [(np.nanstd(XDummy_cluster1_samples.ix[:,colName]),np.nanstd(XDummy_cluster2_samples.ix[:,colName]),np.nanstd(XDummy_cluster3_samples.ix[:,colName]),np.nanstd(XDummy_cluster4_samples.ix[:,colName])) for colName in temp_sorted.ix[:featureNSelect-1,'feature'].tolist()]
separation_mean_df = pd.DataFrame(separation_mean,columns=['mean_cluster_1','mean_cluster_2','mean_cluster_3','mean_cluster_4'])
separation_median_df = pd.DataFrame(separation_median,columns=['median_cluster_1','median_cluster_2','median_cluster_3','median_cluster_4'])
separation_std_df = pd.DataFrame(separation_std,columns=['std_cluster_1','std_cluster_2','std_cluster_3','std_cluster_4'])
(pd.concat([temp_sorted.ix[:featureNSelect-1,'feature'],separation_median_df,separation_mean_df,separation_std_df],axis=1)).to_csv('4_cluster_top_variable_median_mean_std.csv')

ipdb.set_trace()
'''

'''
######## forward selection CLUSTERING
featureSelectedList = []
featureRemainList = list(XForClustering.columns)
iter_i=1
best = 0
bestList = []
k=4
while iter_i <= 5:
    best = 0
    featureWin = None
    startTime = time.time()
    tempFeature = featureSelectedList
    for itemIndex,featureItem in enumerate(featureRemainList):
        tempFeature.append(featureItem)
        clusterData = XForClustering[tempFeature]
        Z = linkage(clusterData,method='ward',metric='euclidean')
        clusters = fcluster(Z,k,criterion='maxclust')
        if len(set(clusters))>=2:
            co = silhouette_score(clusterData,clusters,metric='euclidean')
            #co,coph_dists = cophenet(Z,pdist(clusterData))
            if co>best:
                best = co
                featureWin = featureItem
        if (itemIndex+1)%20==0:
            print 'feature %d done' %(itemIndex+1) 
        tempFeature.pop()
    featureSelectedList.append(featureWin)
    featureRemainList.remove(featureWin)
    bestList.append(best)
    print 'finish iteration %d, cost time %d' %(iter_i,time.time()-startTime)
    print 'cophenetic distances list is:'
    print bestList
    print 'features added are:'
    print featureSelectedList
    iter_i+=1
    ipdb.set_trace()
    
ipdb.set_trace()
'''
    

