# Aquí creamos y entrenamos a multiples nuevos datos
import cv2
import os
import numpy as np

# Directorio donde se almacenarán las imágenes del nuevo usuario
dir_base_datos = 'base_datos'  # Cambia esto por el nombre de tu directorio de base de datos

def capturar_imagenes(nombre_usuario):
    # Crear un directorio para el nuevo usuario
    dir_usuario = os.path.join(dir_base_datos, nombre_usuario)
    os.makedirs(dir_usuario, exist_ok=True)

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    # Capturar múltiples imágenes del usuario
    contador = 0
    while contador < 100:  # Puedes ajustar la cantidad de imágenes a capturar
        ret, frame = cap.read()

        # Mostrar el fotograma actual
        cv2.imshow('Capturando Imágenes', frame)

        # Guardar la imagen capturada en el directorio del usuario
        img_nombre = f'{dir_usuario}/imagen_{contador}.jpg'
        cv2.imwrite(img_nombre, frame)

        contador += 1

        # Esperar a que se presione la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

def actualizar_base_datos(nombre_usuario):
    # Llamar a la función para capturar imágenes del nuevo usuario
    capturar_imagenes(nombre_usuario)

    # Mostrar mensaje de éxito
    print(f'Imágenes de {nombre_usuario} guardadas en la base de datos.')

def verificar_base_datos(nombre_usuario):
    dir_usuario = os.path.join(dir_base_datos, nombre_usuario)
    if os.path.exists(dir_usuario):
        print(f'{nombre_usuario} existe en la base de datos.')
    else:
        print(f'{nombre_usuario} no existe en la base de datos.')

def entrenar_modelo():
    # Inicializar variables para almacenar imágenes y etiquetas
    imagenes = []
    etiquetas = []

    # Iterar sobre los usuarios en la base de datos
    for nombre_usuario in os.listdir(dir_base_datos):
        if not os.path.isdir(os.path.join(dir_base_datos, nombre_usuario)):
            continue

        # Obtener el directorio del usuario actual
        dir_usuario = os.path.join(dir_base_datos, nombre_usuario)

        # Iterar sobre las imágenes del usuario y cargarlas
        for file in os.listdir(dir_usuario):
            if file.endswith("jpg"):
                path = os.path.join(dir_usuario, file)
                imagen = cv2.imread(path, 0)  # Leer en escala de grises
                if imagen is not None:
                    imagenes.append(imagen)
                    # Convertir el nombre de usuario a un número (representación de etiqueta)
                    etiquetas.append(hash(nombre_usuario) % ((2 ** 31) - 1))

    # Crear el clasificador LBPH y entrenarlo con las imágenes y etiquetas
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(imagenes, np.array(etiquetas, dtype=np.int32))  # Asegúrate de que las etiquetas sean np.int32

    # Guardar el modelo entrenado
    recognizer.save('modelo_entrenado.yml')
    print('Modelo entrenado exitosamente.')

# Lógica principal del programa
def main():
    while True:
        print("\n--- Menú de Reconocimiento Facial ---")
        print("1. Registrar Nuevo Usuario")
        print("2. Verificar si ya estás en la base de datos")
        print("3. Entrenar Modelo")
        print("4. Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            nombre_nuevo_usuario = input("Ingresa tu nombre para registrarte en la base de datos: ")
            actualizar_base_datos(nombre_nuevo_usuario)
        elif opcion == "2":
            nombre_usuario_existente = input("Ingresa tu nombre para verificar si estás en la base de datos: ")
            verificar_base_datos(nombre_usuario_existente)
        elif opcion == "3":
            entrenar_modelo()
        elif opcion == "4":
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida. Por favor, selecciona una opción válida.")

# Ejecutar la lógica principal
if __name__ == "__main__":
    main()
