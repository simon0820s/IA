import math
import matplotlib.pyplot as plt
import numpy as np

def run(model,train_data,test_data,class_names):
    BATCH_SIZE=32

    #entregar datos de manera aleatoria, a shuffle se le especifica la cantidad de datos y se define un tamaño de lote
    train_data=train_data.repeat().shuffle(6000).batch(BATCH_SIZE)

    test_data=test_data.batch(BATCH_SIZE)#a los datos de testing solo se les define el tamaño de lote ya que no requieren aleatoriedad

    print("[INFO] generando entrenamiento")
    
    binnacle=model.fit( #se genera el entrenamiento con la funcion fit
        train_data,#datos de entrenamiento
        batch_size=32,
        epochs=5,#iteraciones
        steps_per_epoch=math.ceil(60000/32)#se divide el total de datos entre el tamaño de lote
    )
    loss_graph(binnacle)

def loss_graph(binnacle):
    plt.xlabel("# Epoch")
    plt.ylabel("Loss")
    plt.plot(binnacle.history["loss"])
    plt.savefig(fname='./img_clasificator/train_plot')
    plt.show()