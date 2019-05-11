import os
from PIL import Image
import numpy as np
import sklearn 
from sklearn import model_selection
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPooling2D,SimpleRNN,Embedding,TimeDistributed
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

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

np.random.seed(2)
model = Sequential()

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

#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)

X_train4D = X_train.reshape(604,1,28,28,1).astype('float32')
X_test4D = X_test.reshape(152,1,28,28,1).astype('float32')

X_train4D_normalize = X_train4D / 255
X_test4D_normalize = X_test4D / 255

y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

model.add(TimeDistributed(Conv2D(filters=64,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu')))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
#model.add(TimeDistributed(Conv2D(filters=64,kernel_size=(5,5),activation='relu')))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
#model.add(TimeDistributed(Conv2D(filters=128,kernel_size=(3,3),activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

model.add(TimeDistributed(Flatten()))
#model.add(TimeDistributed(Dropout(0.2)))
#model.add(Embedding(output_dim=28, input_dim=529,input_length=1600))
model.add(TimeDistributed(Dropout(0.3)))
model.add(SimpleRNN(units=32))
model.add(Dense(2,activation='sigmoid'))

#model.compile(loss='binary_crossentropy',optimizer='adam', metrics=[f1])
#model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=[auc])


train_history = model.fit(x=X_train4D_normalize,y=y_trainOneHot, validation_data=(X_test4D_normalize, y_testOneHot),epochs=3000, batch_size=50,verbose=2)

print(model.summary())

#show_train_history(train_history,'f1','val_f1')
#show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'auc','val_auc') 


#scores = model.evaluate(X_test4D_normalize, y_testOneHot)
#print("测试数据集的准确率是：",scores[1])




