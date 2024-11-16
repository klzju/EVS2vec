from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import json
import random

from sklearn.manifold import TSNE


if __name__ == '__main__':
    maxlen = 512
    model_name = 'Att-GRU'
    time = '02241657'

    model = load_model(f"model/{model_name}_{time}/base_network.h5")
    model.summary()

    with open('Test.json') as f:
        TestData = json.load(f)
    Categories = [title for title in list(TestData.keys()) if len(title)<30]

    title_num = 10
    sample_num = 100
    SelectedTitles = random.sample(Categories,k=title_num)

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

    decomposer = TSNE(n_components=2)
    X_d = decomposer.fit_transform(X_output)
    print(X_d.shape)

    sns.set()
    plt.figure(figsize=(6,4),dpi=400)

    for title in SelectedTitles:
        points = X_d[y==title2idx[title]]
        sns.scatterplot(points[:,0],points[:,1],s=5,label=title)
        # plt.scatter(points[:,0],points[:,1],s=3,label=title)
    plt.title(f"VideoTitles:{title_num} SampleNum:{sample_num}")
    plt.legend(
        loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.,
        title='Video Title',
        prop = {'size':7.5}
    )
    plt.show()

    stream_num = 30
    threshold = 0.30835
    DrawData = []
    Fingerprints = []
    result = []
    for title in SelectedTitles:
        EmbData = X_output[y==title2idx[title]]
        title_samples = random.sample(EmbData.tolist(), k=stream_num)
        DrawData.extend(title_samples)
        fp = np.mean(EmbData, axis=0)
        Fingerprints.append(fp.tolist())
        distances = np.sqrt(np.square(np.array(EmbData) - fp).sum(axis=1))
        print(title)
        distance_max, distance_min, distance_mean = distances.max(), distances.min(), distances.mean()
        percentage = np.mean(distances < threshold)
        result.append({'title':title,'percentage': percentage, 'd_min': distance_min, 'd_max': distance_max, 'd_mean': distance_mean})
        print(percentage, f"{distance_mean:4f}", f"{distance_max:.4f}", f"{distance_min:4f}")

    pd.DataFrame(result).to_csv('fingerprint.csv')

    v_max, v_min = np.max(DrawData), np.min(DrawData)
    sns.heatmap(DrawData, cmap='coolwarm', vmin=v_min, vmax=v_max)
    plt.yticks(np.arange(stream_num / 2, title_num * stream_num, stream_num), SelectedTitles, fontsize=7)
    plt.ylabel('Video Titles')
    plt.xlabel('Embedding Dimension')
    plt.title('Embedding Results of Encrypted Video Streams')
    plt.show()

    sns.heatmap(Fingerprints, cmap='coolwarm', vmin=v_min, vmax=v_max, annot=True, fmt=".2f", annot_kws={"fontsize": 8})
    plt.yticks(np.arange(0.5, title_num, 1), SelectedTitles, fontsize=7, rotation=0)
    plt.ylabel('Video Titles')
    plt.xlabel('Embedding Dimension')
    plt.title('Fingerprint of Encrypted Video Stream')
    plt.show()
    #


