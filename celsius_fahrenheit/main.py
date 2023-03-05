import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def main():
    #recolectando datos de entrenamiento
    celsius,fahrenheit=get_input()
    
    #creando una capas
    print("[INFO] creando capas")
    # cap=tf.keras.layers.Dense(
    #     units=1,#cantidad de neuronas
    # input_shape=[1],#capa de entrada con 1 neurona
    # )
    hidden1=tf.keras.layers.Dense(units=3,input_shape=[1])
    hidden2=tf.keras.layers.Dense(units=3,input_shape=[1])
    output=tf.keras.layers.Dense(units=1)

    #creando modelo secuencial
    print("[INFO] creando modelo")

    model=tf.keras.Sequential([hidden1,hidden2,output])#indica la capa
    model.summary() #se imprime el modelo y su estructura

    #compilando el modelo
    print("[INFO] compilando modelo")

    model.compile(
    optimizer=tf.optimizers.Adam(0.01),#se define el optimizador del modelo y su tasa de aprendizaje
    loss='mean_squared_error'#se define la funcion de perdida
    )

    #entrenamiento
    print("[INFO] comenzando el entrenamiento")

    binnacle=model.fit( #se empieza el entrenamiento y se guardan los resultados en la variable historial
        celsius, #se define dato de entrada
        fahrenheit, #se define dato de entrada
        epochs=350 #se define el numero de iteraciones
    )
    print("[INFO] modelo entrenado!!")

    #guardando modelo 
    print("[INFO] guardando modelo")

    model.save('./celsius_fahrenheit/trained_model.h5') #se fuarda y le ponemos extencion h5 para que no entregue una carpeta

    #ver resultados del etrenamiento
    print("[INFO] creando tabla de resultados")

    plt.xlabel("iterations")#crendo label x
    plt.ylabel("loss")#creando label y
    plt.plot(binnacle.history["loss"]) #definiendo los datos a utilizar
    plt.savefig(fname="./celsius_fahrenheit/train_plot")#guardando grafica
    plt.show() #mostrando la grafica

    #Realizando una prediccion con el modelo guardado
    print("[INFO] cargando modelo")

    model_saved=tf.keras.models.load_model('./celsius_fahrenheit/trained_model.h5')
    model_saved.summary()

    print("[INFO] realizando prediccion")

    resultado=model_saved.predict([100.0])#guardando la prediccion de 100.0 grados en una variable resultado
    print(f"el resultado es => {resultado} Fahrenheit")

    #viendo como quedaron las variables internas del modelo
    print("[INFO] cargando variables internas del modelo")
    print(hidden1.get_weights())
    print(hidden2.get_weights())
    print(output.get_weights())


def get_input():
    celsius=np.array([-40,-10,0,8,15,22,38,],dtype=float)
    fahrenheit=np.array([-40,14,32,46,59,72,100],dtype=float)

    return celsius,fahrenheit


if __name__=='__main__':
    main()