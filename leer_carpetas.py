import os

# Ruta de la carpeta principal
ruta_principal = './base_datos'  # Reemplaza con la ruta correcta

# Obtener la lista de subdirectorios en la carpeta principal
subcarpetas = [nombre for nombre in os.listdir(ruta_principal) if os.path.isdir(os.path.join(ruta_principal, nombre))]

# Imprimir los nombres de las subcarpetas
for subcarpeta in subcarpetas:
    print(subcarpeta)
