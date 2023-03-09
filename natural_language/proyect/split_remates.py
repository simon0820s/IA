import re
import get_data


def run():
    text=get_data.run("./Ejemplo.txt")
    print(text)

if __name__=='__main__':
    run()