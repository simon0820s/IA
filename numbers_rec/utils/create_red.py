import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def run(X_train,Y_train):
    #Reiniciar la semilla del generador aleatorio para reproducir el entrenamiento
    np.random.seed(1)

    #Definiendo el tamaño de la entrada y la salida
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    #Usando sequential se creara el contenedor de la red
    model=Sequential()

    """Se agregan la cantidad de neuronas que se van a 
    utilizar en el modelo: 15 se  agrega el la cantidad
    de datos= input_dim y la funcion de activacion"""
    model.add( Dense(15, input_dim=input_dim, activation='relu'))
    
    """Se entrega el tamaño de la salida y la 
    funcion de activacion"""
    model.add( Dense(output_dim, activation='softmax'))

    #se imprime el modelo usando summary
    model.summary()

    return model


