from utils import *
def main():
    train_data,test_data,class_names=get_data.run()#creando variables con los datos obtenidos

    show_imgs.run(train_data,class_names) #se muestra una imagen de ejemplo

    model=create_model.run()#se crea el modelo

    train.run(model,train_data,test_data,class_names)#entrenamiento del modelo

if __name__=='__main__':
    main()