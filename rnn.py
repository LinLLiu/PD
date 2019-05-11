import numpy as np
np.random.seed(1337)  
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
import os
from PIL import Image
from sklearn import model_selection
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf

# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
TIME_STEPS = 28     
INPUT_SIZE = 28    
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 2
CELL_SIZE = 50
LR = 0.0001
 
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.show()

def read_image(imageName):
    im = Image.open(imageName).convert('L')
    data = np.array(im)
    return data

images = []
labels = []
# 读取在datasets里面有几个文件夹 
test = os.listdir('C:/work/keras-test/datasets') 
# 把文件夹里面的图片和其对应的文件夹的名字也就是对应的字 
for testPath in test: 
    for fn in os.listdir(os.path.join('C:/work/keras-test/datasets', testPath)): 
        if fn.endswith('.jpg'): 
            fd = os.path.join('C:/work/keras-test/datasets', testPath, fn) 
            images.append(read_image(fd)) 
            labels.append(testPath) 
X = np.array(images)
y = np.array(list(map(int, labels)))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=30)
 
# data pre-processing
X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)
 
# build RNN model
model = Sequential()
 
# RNN cell
model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       
    output_dim=CELL_SIZE,
    unroll=True,
))
 
# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('sigmoid'))
 
# optimizer
adam = Adam(LR)


#model.compile(optimizer=adam,loss='binary_crossentropy', metrics=[f1])
#model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=[auc])

train_history = model.fit(x=X_train,y=y_train, validation_data=(X_test, y_test),
                          epochs=2000, batch_size=50,verbose=2)

#show_train_history(train_history,'f1','val_f1')
#show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'auc','val_auc')

#scores = model.evaluate(X_test, y_test)
#print("测试数据集的准确率是：",scores[1])
"""
# training
for step in range(10001):
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
 
    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

"""
