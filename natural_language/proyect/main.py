import get_data
import stop
import model
import dictionary

def run():
    text=get_data.run("./Ejemplo.txt")
    text=stop.run(text)
    extract=model.run(text)
    dictionary.run(extract)

if __name__=='__main__':
    run()