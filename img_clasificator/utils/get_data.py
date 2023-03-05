import tensorflow_datasets as tfds
import tensorflow as tf
def run():
    print("[INFO] load data")
    #guardando la descarga de el data set en 2 variables, una para los datos y otra para los metadatos
    data,metadata=tfds.load('fashion_mnist',as_supervised=True,with_info=True)

    print("[INFO] preparing data")
    #este set de datos viene separado por lo tanto solo queda almacenar las particiones en 2 variables
    train_data,test_data=data['train'],data['test']

    #almacenando los nombres de las clases en una variable
    class_names=metadata.features['label'].names
    
    #normalizar los datos de entrada teniendo valores de 0-1 en vez de 0-255 unicamente se necesita una division
    train_data=train_data.map(normalize)
    test_data=test_data.map(normalize)

    #pasamos los datos a cache, para que sea mas rapido ya que los guarda en memoria en lugar de disco
    train_data=train_data.cache()
    test_data=test_data.cache()

    print("[INFO] data ready!!")

    return train_data,test_data,class_names

def normalize(imgs,labels):
    imgs=tf.cast(imgs,tf.float32)#se cambia el formato a uno valido
    imgs/=255 #se aplica la division

    return imgs,labels