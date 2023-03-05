import numpy as np
import random
import matplotlib.pyplot as plt

def run(model,X_test,Y_test):
    #realizando la prediccion
    predict=model.predict(X_test)
    #Creando la varible Y_pred que contiene las clases
    Y_pred=np.argmax(predict,axis=1)
    Graph(X_test,Y_test,Y_pred)


def Graph(X_test,Y_test,Y_pred):
    #Definir atributos de la grafica

    #Seleccionando un numero aleatorio
    id_img=np.random.randint(0,X_test.shape[0],1)
    #Usando el aleatorio como seleccionador random
    idx=id_img

    img = X_test[idx,:].reshape(28,28)
    cat_original = np.argmax(Y_test[idx,:])
    cat_prediccion = Y_pred[idx]

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('"{}" clasificado como "{}"'.format(cat_original,cat_prediccion))
    plt.suptitle('Ejemplos de clasificación en el set de validación')
    plt.show()