from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans,SpectralClustering,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import completeness_score,homogeneity_score,v_measure_score,adjusted_rand_score,adjusted_mutual_info_score,fowlkes_mallows_score

import numpy as np
import pandas as pd
import json
import random

def clustering_ConfusionMatrix(labels_true, labels_pred):
    labels_true=np.array(labels_true)
    labels_pred=np.array(labels_pred)
    if labels_true.shape!=labels_pred.shape:
        raise Exception("Error: y_true and y_pred are not same shape.")
    label=np.unique(labels_pred).tolist()
    label_=[]
    for i in np.unique(labels_pred):
        i_=np.bincount(labels_true[np.where(labels_pred == i)[0]]).argmax()
        label_.append(i_)
    if len(label)!=len(label_):
        raise Exception("Bad Clustering Result.")
    label_tran=dict(zip(label,label_))
    labels_pred_=np.array([label_tran[i] for i in labels_pred])
    return confusion_matrix(y_true=labels_true,y_pred=labels_pred_)


if __name__ == '__main__':
    maxlen = 512
    model_name = 'Att-GRU'
    time = '02241657'

    model = load_model(f"model/{model_name}_{time}/base_network.h5")
    model.summary()

    with open('Test.json') as f:
        TestData = json.load(f)
    Categories = list(TestData.keys())

    title_num = 50
    sample_num = 100
    # SelectedTitles = random.sample(Categories,k=title_num)
    SelectedTitles = Categories

    title2idx = dict(zip(SelectedTitles, range(len(SelectedTitles))))
    idx2title = dict(zip(range(len(SelectedTitles)), SelectedTitles))

    X,y = [],[]
    for title in SelectedTitles:
        title_streams = TestData[title]
        if sample_num>len(title_streams):
            X += TestData[title]
            y += [title2idx[title]]*len(TestData[title])
        else:
            X += random.sample(TestData[title],k=sample_num)
            y += [title2idx[title]] * sample_num
    X = pad_sequences(X, maxlen=maxlen, padding='post', value=-1, dtype=np.float)
    y = np.array(y)
    print(X.shape,y.shape)

    X_output = model.predict(X)

    Models = {
        'K-means':KMeans(n_clusters=title_num),
        'Spectral':SpectralClustering(n_clusters=title_num),
        'Agglomerative': AgglomerativeClustering(n_clusters=title_num),
        'DBSCAN': DBSCAN(eps=0.035),  # 0.035
        'GMM': GaussianMixture(n_components=title_num),
    }
    result = {}
    for alg in Models:
        print(alg)
        model = Models[alg]
        y_ = model.fit_predict(X=X_output)

        h = homogeneity_score(labels_pred=y_, labels_true=y)
        c = completeness_score(labels_pred=y_,labels_true=y)
        v = v_measure_score(labels_pred=y_,labels_true=y)
        ari = adjusted_rand_score(labels_pred=y_, labels_true=y)
        ami = adjusted_mutual_info_score(labels_pred=y_, labels_true=y)
        fm = fowlkes_mallows_score(labels_pred=y_, labels_true=y)
        print(f"h={h:.5f} c={c:.5f} V_measure={v:.5f}")
        print(f"ARI={ari:.5f} AMI={ami:.5f} FM={fm:.5f}")
        result[alg] = {'h': h, 'c': c, 'V_1': v, 'ARI': ari, 'AMI': ami, 'FM': fm}

        ConfusionMatrix = clustering_ConfusionMatrix(labels_pred=y_,labels_true=y,)
        ax = sns.heatmap(ConfusionMatrix, vmin=0, vmax=ConfusionMatrix.max(), cmap='Greens')
        plt.title(f"{alg}")
        # plt.savefig(f"clustering/{alg}.png")
        plt.show()
    pd.DataFrame(result).to_csv('clustering/Result_clustering.csv')
