import cv2
import numpy as np
from flask import Flask,request,jsonify
from base64 import b64decode
import re
import pytesseract

app = Flask(__name__)

@app.route('/', methods=["POST"])
def upload():

    #get img
    data=request.get_json()
    b64_img=data.get("image")
    text=generate_text(b64_img)

    return jsonify({"result":text})

def get_image(function):
    def wrapper(b64_img):

        #apply image transformations
        b64_img=re.sub('^data:image/.+;base64,', '', b64_img)
        b64_img=np.frombuffer(b64decode(b64_img),dtype=np.uint8)

        img=cv2.imdecode(b64_img,cv2.IMREAD_UNCHANGED)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("imgs/original_img.jpg",img)

        imgs=[img]
        (h,w)=img.shape[:2]
        center=(w//2,h//2)

        angle=lambda i:int((i*10))
        namer=lambda angle:f"imgs/img{str(angle)}.jpg"

        for i in range(1,37):
            rotation_matrix=cv2.getRotationMatrix2D(center,angle(i),scale=1.0)

            rotated_img=cv2.warpAffine(img,rotation_matrix,(w,h))

            cv2.imwrite(namer(angle(i)),rotated_img)
            imgs.append(rotated_img)

        text=function(imgs)

        return text

    return wrapper

@get_image
def generate_text(imgs):

    def writer(text):
        d=open("document.txt","w")
        d.write(text)
        d.close()

    to_string=lambda img:pytesseract.image_to_string(img)

    text=""
    
    for img in imgs:

        result=to_string(img)

        if result != None:
            text+="\n\n"+result

    writer(text)

    return text
    
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')