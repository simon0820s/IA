from keras.datasets import mnist
from utils import *
def main():
    
    #cargando y separando categoricamente el dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    #creando la tabla las imagenes selecciónadas de manera aleatoria
    #show_mnist.run(x_train,y_train)

    #configurando las imagenes del dataset para poder usarlas como input en la red neuronal
    X_train,X_test,Y_train,Y_test=initializer.run(x_train,x_test,y_train,y_test)

    #Creando la red
    model=create_red.run(X_train=X_train,Y_train=Y_train)

    #Entrenando la red
    train.run(model,X_train,Y_train)

    #Test del modelo y que predicciones puede hacer
    test.run(X_test,Y_test)

    #Realizando predicciónes
    predict.run(model,X_test,Y_test)

#Entry point
if __name__=='__main__':
    main()