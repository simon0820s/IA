import tensorflow as tf
import logging
import tensorflow_hub as hub
def run():
    logging.info("creando modelo")

    model=create()
    model=compile(model)
    model.save('./model/models/created_model.hdf5')

    logging.info("modelo guardado exitosamente")

def create():
    logging.info("creando capas del modelo")
    
    base=tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))

    for layer in base.layers:
        layer.trainable=False

    model=tf.keras.Sequential([
        #capas ocultas
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),

        #capa de salida binaria
        tf.keras.layers.Dense(units=100,kernel_regularizer=tf.keras.regularizers.l2(1e-5),activation='swish'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(units=50,kernel_regularizer=tf.keras.regularizers.l1(1e-5),activation='swish'),
        tf.keras.layers.Dense(units=1,activation='sigmoid')
    ])
    
    model.summary()
    logging.info("modelo creado")

    return model

def compile(model):
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    logging.info("modelo compilado")
    
    return model
    
if __name__=='__main__':
    run()