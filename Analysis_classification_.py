from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import json
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from xgboost import XGBClassifier

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    Models = {
        'KNN':KNeighborsClassifier(n_neighbors=1),
        'GaussianNB':GaussianNB(),
        'RidgeRegression':RidgeClassifier(),
        'SVM': SVC(),
        'DecisionTree':DecisionTreeClassifier(),
        'RandomForest':RandomForestClassifier(),
        'XGBoost':XGBClassifier()
    }
    Result = {}
    for alg in Models:
        model = Models[alg]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # print(classification_report(y_pred=y_pred,y_true=y_test))
        acc = accuracy_score(y_pred=y_pred,y_true=y_test)
        print(alg,acc)
        Result[alg]=acc
        ConfusionMatrix = confusion_matrix(y_pred=y_pred,y_true=y_test)
        sns.heatmap(ConfusionMatrix, vmin=0, vmax=ConfusionMatrix.max(), cmap='Greens')
        plt.title(f"{alg}")
        plt.show()
        Report = classification_report(
            y_pred=y_pred,y_true=y_test,
            digits=4,output_dict=True,target_names=SelectedTitles,
        )
        print(Report)
        df = pd.DataFrame(Report).transpose()
        df.to_csv(f'classification/{alg}.csv')

