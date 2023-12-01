import os

def obtener_ids():
    ruta_base_datos = './base_datos'  # Ruta de la carpeta base_datos
    ids = []

    # Recorrer las carpetas y subcarpetas de base_datos
    for root, dirs, files in os.walk(ruta_base_datos):
        # Filtrar solo los archivos de imagen (puedes agregar más extensiones si es necesario)
        imagenes = [f for f in files if f.endswith(('png', 'jpg', 'jpeg'))]

        # Imprimir los nombres de archivo para inspeccionar su estructura
        print("Archivos en", root, ":", imagenes)

        # Extraer el ID de cada imagen y agregarlo a la lista de IDs
        for imagen in imagenes:
            # Agrega aquí la lógica para extraer el ID de acuerdo con la estructura de tus nombres de archivo
            # Ejemplo: 
            id_ = imagen.split('_')[0]
            ids.append(id_)

    # Eliminar duplicados y devolver la lista de IDs
    ids = list(set(ids))
    return ids

# Obtener los IDs de los rostros almacenados
lista_ids = obtener_ids()
print("IDs de rostros almacenados:", lista_ids)
