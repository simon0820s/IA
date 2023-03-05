from utils import *
from keras.utils import np_utils
import numpy as np

def run(x_train,x_test,y_train,y_test):
    
    """se realiza un reshape para modificar
    las dimenciones de la grafica debido a que
    la red neuronal solo resive como input
    vectores, se cambia de u
    show_mnist.run(x_train,y_train)na matriz tridimencional
    a una bidimencional"""
    X_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    X_test = np.reshape( x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

    """se debe de garantizar la intensidad de los
    pixeles por lo tanto se divide la imagen entre
    las tonalidades de el dataset en este caso 255.0
    para tenerlo en un rango entre 0-1"""
    X_train = X_train/255.0
    X_test = X_test/255.0

    """las categorias se deben de separar a un formato
    one hot usando binarios en un arreglo por ej:
    la categoria 3=[0,0,0,1] esto se implementa usando
    np_utils de keras"""
    nclasses=10
    Y_train = np_utils.to_categorical(y_train,nclasses)
    Y_test = np_utils.to_categorical(y_test,nclasses)

    return X_train,X_test,Y_train,Y_test