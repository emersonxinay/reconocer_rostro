import cv2
import os
import numpy as np

def cargar_nombres(dir_base_datos):
    nombres = {}
    idx = 0

    for root, dirs, files in os.walk(dir_base_datos):
        for dir_name in dirs:
            nombres[idx] = dir_name
            idx += 1

    return nombres

def reconocer_rostros():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('modelo_entrenado.yml')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    dir_base_datos = 'base_de_datos'
    nombres = cargar_nombres(dir_base_datos)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            nombre = nombres.get(id_, "Desconocido")

            cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Reconocimiento Facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

reconocer_rostros()
