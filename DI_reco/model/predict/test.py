import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils import paths
import tensorflow_hub as hub
logging.basicConfig(level=logging.INFO)
def run():

    PATH="./predict/imgs" #path de la imagen
    imgs=get_img(PATH) #funcion para recolectar la imagen

    model=tf.keras.models.load_model("./models/trained_model.hdf5",custom_objects={'KerasLayer':hub.KerasLayer}) #carga del modelo entrenado

    result=model.predict(imgs)
    result_1=result[0,1]
    result_2=result[1,1]

    graph(result_1,result_2,imgs) #graficar el resultado

def get_img(PATH):
    logging.info("cargando imagenes")
    imgs=[]
    img_paths=list(paths.list_images(PATH))

    for i in range(0,len(img_paths)):
        img=cv2.imread(img_paths[i])
        img=cv2.resize(img,(224,224))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgs.append(img)
    imgs=np.array(imgs).astype(float)/255

    return imgs

def graph(result_1,result_2,imgs):
    logging.info("graficando resultados")
        
    fig=plt.figure(figsize=(5,5))

    plt.subplot(1,2,1)
    plt.imshow(imgs[0])
    plt.axis('off')
    plt.title(f"categoria: {result_1}")

    plt.subplot(1,2,2)
    plt.imshow(imgs[1])
    plt.axis('off')
    plt.title(f"categoria: {result_2}")

    plt.show()
    
   
if __name__=='__main__':
    run()