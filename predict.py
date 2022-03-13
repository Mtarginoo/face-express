import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

cascade_faces = "/app/Material/Material/haarcascade_frontalface_default.xml"
caminho_modelo = "/app/Material/Material/modelo_01_expressoes.h5"
#cascade_faces = "/home/mtarginoo/Personal_Projects/faceExpress/Material/Material/haarcascade_frontalface_default.xml"
#caminho_modelo = "/home/mtarginoo/Personal_Projects/faceExpress/Material/Material/modelo_01_expressoes.h5"

face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile = False)
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]


def predict(imagem):
    original = imagem.copy()
    faces = face_detection.detectMultiScale(original, scaleFactor = 1.1, minNeighbors = 3, minSize = (20,20))
    cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    predictions = {}

    for i, (x, y, w, h) in enumerate(faces):
        # Extraindo a região de interesse (ROI)
        roi = cinza[y:y + h, x:x + w]

        #redimensionando a imagem
        roi = cv2.resize(roi, (48, 48))

        #Normalizando
        roi = roi.astype("float")/255
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)

        # Previsões
        preds = classificador_emocoes.predict(roi)[0]

        # Emoção detectada
        emotion_probability = np.max(preds)
        #print(emotion_probability)

        #print(preds.argmax())
        label = expressoes[preds.argmax()]

        # Mostra resultado na tela para o rosto
        cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

        predictions.setdefault(str(i+1), {})['label'] = label
        predictions.setdefault(str(i+1), {}).setdefault('face', {})['x_box'] = str(x)
        predictions.setdefault(str(i+1), {}).setdefault('face', {})['y_box'] = str(y)
        predictions.setdefault(str(i+1), {}).setdefault('face', {})['w_box'] = str(w)
        predictions.setdefault(str(i+1), {}).setdefault('face', {})['h_box'] = str(h)
        predictions.setdefault(str(i+1), {})['score'] = str(emotion_probability)

    return predictions
    
