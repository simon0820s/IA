import tensorflow as tf
import logging
from tensorflow.keras import regularizers

def run():
    logging.info("creando modelo")

    model=create()
    model.save('./model/models/created_model.hdf5')

    logging.info("modelo guardado exitosamente")

def create():
    logging.info("creando capas del modelo")

    model=tf.keras.models.Sequential([

        #seccion convolucional
        tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(336,336,3)),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(48,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(3,3),

        tf.keras.layers.Conv2D(48,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(3,3),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=100,kernel_regularizer=regularizers.l2(1e-5),activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(units=100,kernel_regularizer=regularizers.l2(1e-5),activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(units=1,activation='sigmoid')
    ])
    
    model.summary()
    logging.info("modelo creado")

    return model
    
if __name__=='__main__':
    run()

