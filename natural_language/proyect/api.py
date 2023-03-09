from flask import Flask, request, jsonify
from flask_cors import CORS
import stop
import model
import dictionary

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["POST"])
def upload():
    data = request.get_json()
    text=data.get("text")
    text_ws=stop.run(text)
    extract=model.run(text_ws)
    dty=dictionary.run(extract)
    dty["text"]=text

    return jsonify(dty)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
