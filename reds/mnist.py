import numpy as np
from keras import layers,models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf

#cargando data
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()


model=models.Sequential() #creando modelo

#creando capa densa
model.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))

#compilar
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics='accuracy'
)
model.summary()

#preparando data
x_train=train_data.reshape(60000,28*28)
x_train=x_train.astype('float32')/255

x_test=test_data.reshape(10000,28*28)
x_test=x_test.astype('float32')/255

y_train=to_categorical(train_labels)
y_test=to_categorical(test_labels)


#entrenar la red
model.fit(x_train,y_train,epochs=5,batch_size=128)

#evaluar
result=model.evaluate(x_test,y_test)

print("el acuraccy fue de: "+str(result[1]*100))