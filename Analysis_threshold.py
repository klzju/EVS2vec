import json
import random

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Input
import tensorflow as tf
import numpy as np
import math
import json
import random


def create_pairs(JsonData,maxlen,sample_num=100000):
    X0,X1,y = [],[],[]
    Categories = list(JsonData.keys())
    for i in range(sample_num):
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            title = random.choice(Categories)
            stream0, stream1 = random.sample(JsonData[title], k=2)
        else:
            title0, title1 = random.sample(Categories, k=2)
            stream0 = random.choice(JsonData[title0])
            stream1 = random.choice(JsonData[title1])
        stream0,stream1 = np.array(stream0),np.array(stream1)
        # stream0,stream1 = stream0/(1024*1024),stream1/(1024*1024)

        X0.append(stream0)
        X1.append(stream1)
        y.append(should_get_same_class)
    X0 = pad_sequences(X0,maxlen=maxlen,padding='post',value=-1,dtype=np.float)
    X1 = pad_sequences(X1,maxlen=maxlen,padding='post',value=-1,dtype=np.float)
    return X0,X1,np.array(y,dtype=np.float)


from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def Threshold_Acc(y_pred,y_true,threshold):
    y_pred_=y_pred<threshold
    return np.mean(y_pred_==y_true)

if __name__ == '__main__':
    maxlen = 512
    model_name = 'Att'
    time = '03301846'

    model = load_model(f"model/{model_name}_{time}/base_network.h5")
    model.summary()

    with open('Test.json') as f:
        TestData = json.load(f)
    X0_test, X1_test, y_test = create_pairs(TestData, maxlen=maxlen,sample_num=100000)
    X0_test, X1_test = X0_test[:, :, np.newaxis], X1_test[:, :, np.newaxis]

    start = datetime.now()
    X0_test_d = model.predict(X0_test)
    X1_test_d = model.predict(X1_test)
    end = datetime.now()
    dur = (end-start).total_seconds()
    print(f"TotalTime: {dur}")

    distance = np.sqrt(np.sum(np.square(X0_test_d - X1_test_d), axis=1))
    pos = distance[y_test == 1.]
    neg = distance[y_test == 0.]
    print(pos.mean(),pos.std())
    print(neg.mean(),neg.std())
    fpr, tpr, thresholds = roc_curve(y_test, distance)
    threshold = thresholds[(fpr-tpr).argmax()]
    acc = Threshold_Acc(y_pred=distance,y_true=y_test,threshold=threshold)
    print("Threshold:",threshold)
    print("Accuracy:",acc)

    sns.set()
    sns.distplot(pos,label='same')
    sns.distplot(neg,label='different')
    plt.title(f'{model_name} Distance')
    plt.legend()
    plt.show()

    pass


    # historys = model.fit_generator(
    #     generator=train_generator,
    #     validation_data=({'Input1': X0_test, 'Input2': X1_test}, y_test),
    #     epochs=50
    # )




