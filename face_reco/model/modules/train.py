import matplotlib.pyplot as plt
import tensorflow as tf
import get_data
import logging

logging.basicConfig(level=logging.INFO)

def run():
    logging.info("cargando datos y modelo")
    path="./model/data/"
    gen_train,gen_val=get_data.run(path) #funcion de obtener datos

    model=tf.keras.models.load_model('model/models/created_model.hdf5',compile=False)#funcion de cargar modelo
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    train(model,gen_train,gen_val)
    
def train(model,gen_train,gen_val):

    early=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=40) #callback

    check_point=tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/models/trained_model.hdf5', #ruta
        verbose=1, #print
        monitor='val_accuracy', #monitor
        save_best_only=True) #guardar solo el mejor

    binnacle=model.fit( #entrenamiento del modelo
        gen_train, #datos de entrenamiento con el generador
        batch_size=16, #tamaño de lotes
        epochs=200, #definiendo epocas
        callbacks=[early,check_point], #callback
        validation_data=gen_val, #validacion
        )

    logging.info("[INFO] modelo entrenado y guardado")
    graph(binnacle) #graficar

def graph(binnacle):

    logging.info("graficando datos del entrenamiento")
    
    #definiendo variables
    history_dict = binnacle.history
    train_loss=history_dict['loss'] #train loss
    val_loss=history_dict['val_loss'] #validation loss
    train_accuracy=history_dict['accuracy']
    val_accuracy=history_dict['val_accuracy']
    epoch=range(1,len(train_loss)+1) #epocas


    plt.figure(figsize=(5,5)) #tamaño de la grafica
    plt.plot(epoch,train_loss,'-',label="train_loss") #graficar los datos del train set
    plt.plot(epoch,val_loss,'-',label="validation_loss") #graficar los datos del validation set
    plt.plot(epoch,train_accuracy,'--',label="train_accuracy") #graficar los datos del train set
    plt.plot(epoch,val_accuracy,'--',label="validation_accuracy") #graficar los datos del validation set
    plt.xlabel("epoch") #label x
    plt.ylabel("metrics") #label y
    #mostrar y finalizar
    plt.legend()
    plt.savefig("./train_plot.jpg")
    plt.show()


if __name__=='__main__':
    run()