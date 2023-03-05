import tensorflow as tf
import get_data
import logging
import tensorflow_hub as hub

logging.basicConfig(level=logging.INFO)

def run():
    logging.info("[INFO] cargando data")
    path="./model/data/COL_data"
    gen_train,gen_val=get_data.run(path)#funcion de obtener datos
    model=tf.keras.models.load_model('./model/models/trained_model.hdf5',custom_objects={'KerasLayer':hub.KerasLayer})#funcion de cargar modelo

    logging.info("[INFO] haciendo testing")

    score=model.evaluate(gen_val,verbose=1)#guardando el puntaje del set de validacion
    print('accuracy: {}%'.format((100*round(score[1],2)))) #print de el rendimiento en la funcion de perdida en porcentaje
    print(f"score: {score}") #score en consola

    predictions=model.predict(gen_val)
    print(predictions)
    
    logging.info("[INFO] modelo testeado")
    
if __name__=='__main__':
    run()