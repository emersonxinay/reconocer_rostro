import cv2
import os

def reconocer_rostros():
    # Cargar el modelo entrenado
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('modelo_entrenado.yml')

    # Cargar el clasificador de detección de rostros pre-entrenado
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iniciar la cámara
    cap = cv2.VideoCapture(0)

    # Ruta de la carpeta que contiene las subcarpetas con imágenes de entrenamiento
    ruta_entrenamiento = './base_datos'  # Reemplaza con la ruta correcta
    nombres = {
        770425446: "Emerson",
        534346286: "Yesica"
        # Puedes agregar más IDs y nombres según corresponda
    }

    while True:
        # Capturar fotograma por fotograma
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Para cada rostro detectado, intentar reconocer
        for (x, y, w, h) in faces:
            # Obtener el área de interés (ROI) del rostro detectado
            roi_gray = gray[y:y+h, x:x+w]

            # Realizar el reconocimiento facial
            id_, conf = recognizer.predict(roi_gray)

            # Verificar si la confianza es alta y el ID está en el diccionario de nombres
            if conf < 70 and id_ in nombres:
                nombre = nombres[id_]
            else:
                nombre = "Desconocido"

            # Mostrar el nombre sobre el rostro detectado
            cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Dibujar un rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Mostrar el fotograma con los rostros detectados y nombres (si se reconocieron)
        cv2.imshow('Reconocimiento Facial', frame)

        # Salir del bucle cuando se presione 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para reconocer rostros
reconocer_rostros()
