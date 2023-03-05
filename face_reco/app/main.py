from flask import Flask, request, jsonify
from flask_cors import CORS
#from gevent.pywsgi import WSGIServer
import tensorflow as tf
import cv2
import numpy as np
from base64 import b64decode
import re
from class_p import Prediction

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("trained_model.hdf5")


@app.route('/', methods=["POST"])
def upload():
    data = request.get_json()
    img_b64=data.get("image")
    result = predict(img_b64)
    print(result)
    prediction=Prediction(result=float(result),score=float(result),b64=img_b64)
    return prediction.to_jsonify()

def prepare_img(function):
    print("decoradito")
    def wrapper(img):
        img = re.sub('^data:image/.+;base64,', '', img)
        img = np.frombuffer(b64decode(img), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (336,336))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = np.array([img]).astype(float)/255

        result=function(imgs)

        return result
    
    return wrapper

@prepare_img
def predict(img):
    result = model.predict(img)
    return result


if __name__ == '__main__':
    app.run(debug=True, port=2809, host='0.0.0.0')
    #http_server = WSGIServer(('', 2809), app)
    #http_server.serve_forever()
