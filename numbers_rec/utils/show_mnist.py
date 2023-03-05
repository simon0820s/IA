import matplotlib.pyplot as plt
import numpy as np

def run(x_train,y_train):
    """Resive como argumentos los datos recolectados
    de las imagenes
    """

    #Selecciona elementos random de el data_set
    ids_imgs = np.random.randint(0,x_train.shape[0],16)

    #iterador para crear las graficas una por una
    for i in range(len(ids_imgs)):
        #Definicion de la configuracion de la grafica
        img = x_train[ids_imgs[i],:,:]
        plt.subplot(4,4,i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(y_train[ids_imgs[i]])
    #Define un subtitulo para mejorar la legibilidad
    plt.suptitle('16 imágenes del set MNIST')
    #Finish
    plt.show()