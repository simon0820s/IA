import tensorflow as tf
def run(X_test,Y_test):
    #Los atributos son los datos que el modelo nunca ha visto
    #cargamos el modelo guardado
    print("[INFO] cargando modelo")
    model=tf.keras.models.load_model('./numbers_rec/trained_model.h5')

    #los datos de evaluate se guardan en score
    score=model.evaluate(X_test,Y_test,verbose=0)
    #Se hace un print del score
    print('Precisión en el set de validación: {:.1f}%'.format(100*score[1]))
    