from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import random

# tf.Traffic_Bilibili.Dataset

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


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, JsonData, maxlen, batch_size, size=100):
        self.Categories = list(JsonData.keys())
        self.JsonData = JsonData
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X0, X1, y = [], [], []
        for i in range(self.batch_size):
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                title = random.choice(self.Categories)
                stream0, stream1 = random.sample(self.JsonData[title], k=2)
            else:
                title0, title1 = random.sample(self.Categories, k=2)
                stream0 = random.choice(self.JsonData[title0])
                stream1 = random.choice(self.JsonData[title1])
            stream0, stream1 = np.array(stream0), np.array(stream1)
            X0.append(stream0)
            X1.append(stream1)
            y.append(should_get_same_class)
        X0 = pad_sequences(X0, maxlen=self.maxlen, padding='post', value=-1, dtype=np.float)
        X1 = pad_sequences(X1, maxlen=self.maxlen, padding='post', value=-1, dtype=np.float)
        return (X0, X1), np.array(y, dtype=np.float)

    def on_epoch_end(self):
        """执行完一个`epoch`之后，还可以做一些其他的事情！"""
        return


from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Lambda,Flatten,Dropout,BatchNormalization
from tensorflow.keras.layers import Masking,LSTM,GRU,Bidirectional,concatenate
from tensorflow.keras.layers import Attention,AdditiveAttention,MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns

def LSTM_model(input_shape,RNNcells=128,attention=False):
    input = Input(shape=input_shape)
    x = Masking(mask_value=-1.)(input)
    lstm, state_h, state_c = LSTM(units=RNNcells, return_sequences=True, return_state=True)(x)
    lstm, state_h, state_c = LSTM(units=RNNcells, return_sequences=True, return_state=True)(lstm)
    lstm, state_h, state_c = LSTM(units=RNNcells, return_sequences=True, return_state=True)(lstm)
    if attention:
        att, weight = Attention()(inputs=[lstm, lstm], return_attention_scores=True)
        att = GlobalAveragePooling1D()(att)
        x = concatenate([state_h, att])
    else:
        x = state_h
    x = BatchNormalization()(x)
    x = Dense(units=16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=8)(x)
    return Model(input, x)

def GRU_model(input_shape,RNNcells=128,attention=False):
    input = Input(shape=input_shape)
    x = Masking(mask_value=-1.)(input)
    gru, state = GRU(units=RNNcells,return_sequences=True,return_state=True)(x)
    gru, state = GRU(units=RNNcells, return_sequences=True, return_state=True)(gru)
    gru, state = GRU(units=RNNcells, return_sequences=True, return_state=True)(gru)
    if attention:
        att, weight = Attention()(inputs=[gru, gru], return_attention_scores=True)
        att = GlobalAveragePooling1D()(att)
        x = concatenate([state, att])
    else:
        x = state
    x = BatchNormalization()(x)
    x = Dense(units=16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=8)(x)
    return Model(input, x)

def Attention_model(input_shape):
    input = Input(shape=input_shape)
    x = Masking(mask_value=-1.)(input)
    x, weight = Attention()(inputs=[x,x],return_attention_scores=True)
    # x, weight = MultiHeadAttention(num_heads=2,key_dim=1)(x,x,return_attention_scores=True)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(units=16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=8)(x)
    return Model(input, x)


def create_base_network(input_shape,selected):
    models = {
        'LSTM': LSTM_model(input_shape,RNNcells=32),
        'Att-LSTM':LSTM_model(input_shape,RNNcells=32,attention=True),
        'GRU': GRU_model(input_shape,RNNcells=32),
        'Att-GRU': GRU_model(input_shape,RNNcells=32,attention=True),
        'Att':Attention_model(input_shape),
    }
    return models[selected]

def distance(vects):
    x1, x2 = vects
    sum_square = K.sum(K.square(x1 - x2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

from sklearn.metrics import roc_curve
def contrastive_loss(y_true,y_pred):
    pos_loss = K.square(y_pred)
    neg_loss=K.square(1-y_pred)
    # Q=5
    # pos_loss = 1/Q*K.abs  (y_pred)
    # neg_loss = Q*K.abs(y_pred+1/K.square(y_pred))
    return K.mean(y_true*pos_loss+(1-y_true)*neg_loss)
def accuracy(y_true,y_pred):
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # threshold = thresholds[(fpr-tpr).argmax()]
    threshold=0.5
    return K.mean(K.equal(y_true,K.cast(y_pred<threshold,y_true.dtype)))

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from datetime import datetime
import os

if __name__ == '__main__':
    model_name = 'Att-GRU'
    time = datetime.now().strftime("%m%d%H%M")
    os.makedirs(f"model/{model_name}_{time}")

    maxlen = 512
    base_network = create_base_network(input_shape=(maxlen, 1),selected=model_name)
    input1 = Input(shape=(maxlen, 1), name='Input1')
    input2 = Input(shape=(maxlen, 1), name='Input2')
    x1 = base_network(input1)
    x2 = base_network(input2)
    output = Lambda(distance, output_shape=(16,), name='Output')([x1, x2])

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(
        loss=contrastive_loss,
        optimizer='adam',
        metrics=[accuracy]
    )
    model.summary()
    plot_model(model, to_file=f'model/{model_name}_{time}/model.png', show_shapes=True, )
    plot_model(base_network, to_file=f'model/{model_name}_{time}/base_network_{model_name}.png', show_shapes=True, )

    with open('Train.json') as f:
        TrainData = json.load(f)
    train_generator = DataGenerator(JsonData=TrainData,maxlen=maxlen,batch_size=256,size=100)
    with open('Test.json') as f:
        TestData = json.load(f)
    X0_test, X1_test, y_test = create_pairs(TestData, maxlen=maxlen,sample_num=10000)
    X0_test, X1_test = X0_test[:, :, np.newaxis], X1_test[:, :, np.newaxis]
    # test_generator = DataGenerator(JsonData=TestData,batch_size=128)

    checkpoint = ModelCheckpoint(
        f"model/{model_name}_{time}/model.h5",
        monitor='val_loss', verbose=1,
        save_weights_only=True,
        save_best_only=True, mode='min',
    )
    historys = model.fit_generator(
        generator=train_generator,
        validation_data=({'Input1': X0_test, 'Input2': X1_test}, y_test),
        epochs=200,
        callbacks=[checkpoint]
    )
    pd.DataFrame(historys.history).to_csv(f'model/{model_name}_{time}/history.csv')

    plt.plot(historys.history['loss'],label='loss')
    plt.plot(historys.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    plt.plot(historys.history['accuracy'], label='accuracy')
    plt.plot(historys.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    # 保存最终模型
    model.save_weights(f"model/{model_name}_{time}/model_.h5")
    base_network = model.layers[2]
    base_network.save(f"model/{model_name}_{time}/base_network_.h5")

    # 保存最优模型
    model.load_weights(f"model/{model_name}_{time}/model.h5")
    base_network = model.layers[2]
    base_network.save(f"model/{model_name}_{time}/base_network.h5")


