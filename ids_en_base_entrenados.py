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

# Ruta del modelo entrenado y la carpeta base_datos
modelo_entrenado = 'modelo_entrenado.yml'  # Reemplazar con la ruta correcta
ruta_base_datos = './base_datos'  # Reemplazar con la ruta correcta

# Obtener los IDs de los rostros almacenados por carpeta basados en el modelo entrenado
diccionario_ids_por_carpeta = obtener_ids_entrenados(modelo_entrenado, ruta_base_datos)
print("IDs de rostros almacenados por carpeta:", diccionario_ids_por_carpeta)
