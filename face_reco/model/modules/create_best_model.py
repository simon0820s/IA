import tensorflow as tf
import logging
from tensorflow.keras import regularizers
import keras_tuner as kt
import get_data
import matplotlib.pyplot as plt

def run():
    logging.info("creando modelo")
    path="model/data"
    gen_train,gen_val=get_data.run(path)

    train(gen_train,gen_val)

    logging.info("modelo guardado exitosamente")

def constructor(hp):
    logging.info("creando capas del modelo")
    model=tf.keras.models.Sequential()

    hp_filters=hp.Int("filters",min_value=16,max_value=64,step=16)
    #seccion convolucional
    model.add(tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(224,224,3)))

    model.add(tf.keras.layers.Conv2D(hp_filters,(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(3,3))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(hp_filters,(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(3,3))


    model.add(tf.keras.layers.Flatten())

    #capas convolucionales
    hp_units=hp.Int("units",min_value=25,max_value=150,step=25)

    model.add(tf.keras.layers.Dense(units=hp_units,kernel_regularizer=regularizers.l2(1e-5),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=hp_units,kernel_regularizer=regularizers.l2(1e-5),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    logging.info("modelo compilado")

    return model

def train(gen_train,gen_val):

    early=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4) #callback

    check_point=tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/models/trained_model.hdf5', #ruta
        verbose=1, #print
        monitor='val_accuracy', #monitor
        save_best_only=True) #guardar solo el mejor

    tuner=kt.Hyperband(
        constructor,
        objective="val_accuracy",
        max_epochs=100,
        factor=3,
        directory="model/models",
        project_name="auco-face",
    )
    tuner.search(gen_train,epochs=100,validation_data=gen_val)

    best=tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model=tuner.hypermodel.build(best)
    best_model.save('./model/models/best_model.hdf5')
    best_model.summary()
    binnacle=best_model.fit(
        gen_train,
        epochs=100,
        callbacks=[early,check_point],
        validation_data=gen_val
    )
    best_model.save('./model/models/best_trained_model.hdf5')
    graph(binnacle)

def graph(binnacle):

    logging.info("graficando datos del entrenamiento")
    
    #definiendo variables
    history_dict = binnacle.history
    train_accuracy=history_dict['accuracy'] #train loss
    val_accuracy=history_dict['val_accuracy'] #validation loss
    train_loss=history_dict['loss'] #train loss
    val_loss=history_dict['val_loss'] #validation loss
    epoch=range(1,len(train_loss)+1) #epocas


    plt.figure(figsize=(5,5)) #tamaño de la grafica
    plt.plot(epoch,train_loss,'-',label="train_loss") #graficar los datos del train set
    plt.plot(epoch,val_loss,'-',label="validation_loss") #graficar los datos del validation set
    plt.plot(epoch,train_accuracy,'-',label="train_accuracy") #graficar los datos del train set
    plt.plot(epoch,val_accuracy,'-',label="validation_accuracy") #graficar los datos del validation set
    plt.xlabel("epoch") #label x
    plt.ylabel("metrics") #label y
    #mostrar y finalizar
    plt.legend()
    plt.savefig("./train_plot.jpg")
    plt.show()

if __name__=='__main__':
    run()
