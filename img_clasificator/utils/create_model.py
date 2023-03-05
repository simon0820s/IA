import tensorflow as tf

def run():
    model=create()#crea el modelo
    model=compile(model)#compila el modelo

    print("[INFO] el modelo ha sido creado y compilado")
    model.summary()
    return model
    

def create():
    print("[INFO] creando modelo")

    model=tf.keras.Sequential([#se crea un modelo con una red de tipo secuencial
        tf.keras.layers.Flatten(input_shape=(28,28,1)), #esta capa aplasta la imagen y la lleva a 1D se le especifica el tamaño y solo 1 canal de color

        #2 capas cada una con 50 neuronas y funcion de activacion relu
        tf.keras.layers.Dense(units=50,activation=tf.nn.relu),
        tf.keras.layers.Dense(units=50,activation=tf.nn.relu),

        #1 capa de salida con 10 neuronas, una por categoria y activacion softmax para salida multicategorica
        tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
    ])
    return model

def compile(model):
    model.compile(
        optimizer='adam',#define optimizador
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), #define funcion de perdida
        metrics=['accuracy']
    )
    return model