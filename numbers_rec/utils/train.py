from keras.optimizers import SGD
import matplotlib.pyplot as plt
def run(model,X_train,Y_train):
    #Definir el metodo de optimizacion de memoria
    #Definir tasa de aprendizaje
    sgd=SGD(lr=0.2)

    """Definir la funcion de error: loss
    Definir la metrica de desempeño de la red neuronal:
    precision"""
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #denifir las iteraciones de entrenammiento
    num_epochs=50

    #Definir la cantidad de elementos que se van a tomar
    batch_size=1024

    """Usando fit inicializamos el entrenamiento
    usando los parametros recolectaodos
    y verbose=2 imprime el entrenamiento"""
    history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=2)
    #Crear una grafica con los datos 
    Graph(historia=history)
    model.save("./numbers_rec/trained_model.h5")

def Graph(historia):
    #Definiendo los datos a graficar
    plt.plot(historia.history['loss'])
    #Definiendo titulo
    plt.title('Pérdida vs. iteraciones')
    #Definiendo los axis
    plt.ylabel('Perdida')
    plt.xlabel('Iteración')
    #Guardando grafica
    plt.savefig("./numbers_rec/plot.jpg")
    #Show and finish
    plt.show()