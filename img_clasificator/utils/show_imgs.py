import numpy as np
import matplotlib.pyplot as plt

def run(train_data,class_names):
    print("[INFO] creating graph")
    #crenado un marco para la grafica
    plt.figure(figsize=(5,5))

    #creando un for para tomar 1 imagen del set de datos
    for i,(img,label) in enumerate(train_data.take(6)):

        img=img.numpy().reshape((28,28))#redimencionar la imagen

        #crenado grafica con la imagen seleccionada
        plt.subplot(2,3,i+1)#creando tabla para almacenar imagenes
        plt.xticks([])#eliminando estilos
        plt.yticks([])#eliminando estilos
        plt.grid(False)#eliminando la cuadricula
        plt.imshow(img,cmap=plt.cm.binary)#mostrando la imagen
        plt.xlabel(class_names[label])#etiquetando la imagen

    plt.show()
    