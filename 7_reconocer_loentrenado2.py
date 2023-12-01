import cv2
import os

def obtener_ids_entrenados(modelo_entrenado, ruta_base_datos):
    # Cargar el modelo entrenado
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(modelo_entrenado)

    # Cargar el clasificador de detección de rostros pre-entrenado
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    ids_por_carpeta = {}

    # Recorrer las carpetas y subcarpetas de base_datos
    for root, dirs, files in os.walk(ruta_base_datos):
        # Filtrar solo los archivos de imagen (puedes agregar más extensiones si es necesario)
        imagenes = [f for f in files if f.endswith(('png', 'jpg', 'jpeg'))]

        # Extraer el nombre de la carpeta actual
        nombre_carpeta = os.path.basename(root)

        # Inicializar la lista de IDs para la carpeta actual
        if nombre_carpeta not in ids_por_carpeta:
            ids_por_carpeta[nombre_carpeta] = []

        # Obtener el ID para cada imagen y agregarlo a la lista de IDs de la carpeta actual
        for imagen in imagenes:
            ruta_imagen = os.path.join(root, imagen)
            img_gray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

            # Detectar rostro en la imagen
            faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = img_gray[y:y + h, x:x + w]
                id_, _ = recognizer.predict(roi_gray)

                ids_por_carpeta[nombre_carpeta].append(id_)

    # Eliminar duplicados y devolver el diccionario de IDs por carpeta
    for carpeta, ids in ids_por_carpeta.items():
        ids_por_carpeta[carpeta] = list(set(ids))
    
    return ids_por_carpeta


def reconocer_rostros(modelo_entrenado, ruta_base_datos):
    # Cargar el modelo entrenado
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(modelo_entrenado)

    # Cargar el clasificador de detección de rostros pre-entrenado
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Obtener los IDs de los rostros almacenados por carpeta
    diccionario_ids_por_carpeta = obtener_ids_entrenados(modelo_entrenado, ruta_base_datos)

    # Iniciar la cámara
    cap = cv2.VideoCapture(0)

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

            # Verificar si el ID está en el diccionario de IDs por carpeta
            nombre = "Desconocido"
            for carpeta, ids in diccionario_ids_por_carpeta.items():
                if id_ in ids:
                    nombre = carpeta
                    break

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
    

# Ruta del modelo entrenado y la carpeta base_datos
modelo_entrenado = 'modelo_entrenado.yml'  # Reemplazar con la ruta correcta
ruta_base_datos = './base_datos'  # Reemplazar con la ruta correcta

# Obtener los IDs de los rostros almacenados por carpeta basados en el modelo entrenado
diccionario_ids_por_carpeta = obtener_ids_entrenados(modelo_entrenado, ruta_base_datos)
print("IDs de rostros almacenados por carpeta:", diccionario_ids_por_carpeta)

# Llamar a la función para reconocer rostros
reconocer_rostros(modelo_entrenado, ruta_base_datos)


