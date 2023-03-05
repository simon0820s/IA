import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO) #estableciendo nivel de los mensajes

def run(path):
    logging.info("Obteniendo data")
    
    generator=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, #normalizar
        zoom_range=0.1, #rango de zoom
        width_shift_range=0.1, #movimiento en el eje horizontal
        height_shift_range=0.1, #movimiento en el eje vertical
        fill_mode='nearest', #fondo de imagen
        brightness_range=[0.4,1.5], #rango de brillos
        validation_split=0.1 #20% para pruebas
    )
    gen_train=generator.flow_from_directory(
        path,
        target_size=(336,336),
        batch_size=32,
        shuffle=True,
        class_mode='binary',
        subset="training"
        )

    gen_val=generator.flow_from_directory(
        path,
        target_size=(336,336),
        batch_size=32,
        shuffle=True,
        class_mode='binary',
        subset="validation"
    )
    
    return gen_train,gen_val