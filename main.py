#from distutils.log import debug
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image
import urllib.request
from key_generator.key_generator import generate
import json

from predict import predict
import os

DIRETORIO_UPLOADS = "/app/upload_images"
DIRETORIO_PREDICOES = "/app/predictions_images"
# DIRETORIO_UPLOADS = "/home/mtarginoo/Personal_Projects/faceExpress/upload_images"
# DIRETORIO_PREDICOES = "/home/mtarginoo/Personal_Projects/faceExpress/predictions_images"

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_link', methods=['POST'])
def upload_link():
    link = request.form.get("imgLink")

    key = generate(seed = 1)
    file_name = str(key.get_key()) + ".png"

    urllib.request.urlretrieve(link, DIRETORIO_UPLOADS + "/" + file_name)
    
    imagem = cv2.imread(DIRETORIO_UPLOADS + '/' + file_name)    
    predictions = predict(imagem)

    predictions_json = json.dumps(predictions)

    return predictions_json

    # cv2.imwrite(DIRETORIO_PREDICOES + "/classificada.png", imagem_classificada)

    # return send_from_directory(DIRETORIO_PREDICOES, "classificada.png", as_attachment=False)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    key = generate(seed = 101)
    file_name = str(key.get_key())

    arquivo = request.files.get("image")
    arquivo.save(os.path.join(DIRETORIO_UPLOADS, file_name))

    imagem = cv2.imread(DIRETORIO_UPLOADS + '/' + file_name)    
    predictions = predict(imagem)

    predictions_json = json.dumps(predictions)

    return predictions_json
    # cv2.imwrite(DIRETORIO_PREDICOES + "/classificada.png", imagem_classificada)

    # return send_from_directory(DIRETORIO_PREDICOES, "classificada.png", as_attachment=False)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)