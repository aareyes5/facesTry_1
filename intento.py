import os
from flask import Flask, render_template, request, jsonify, url_for,send_from_directory
import cv2
import numpy as np
from PIL import Image
import io
import dlib
import traceback
import base64 

app = Flask(__name__)

def extraer_puntos_referencia(imagen):
    # Cargar el modelo preentrenado para la detección de puntos faciales
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Convierte la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detecta caras en la imagen
    caras = detector(imagen_gris)

    if not caras:
        raise ValueError("No se encontraron caras en la imagen.")

    # Selecciona la primera cara (puedes ajustar esto según tus necesidades)
    cara = caras[0]

    # Obtiene los puntos faciales
    puntos_faciales = predictor(imagen_gris, cara)

    # Convierte los puntos faciales a un array de NumPy
    puntos_referencia = np.array([[p.x, p.y] for p in puntos_faciales.parts()])

    # Excluir puntos faciales relacionados con los ojos
    # Puedes ajustar estos índices según las necesidades específicas de tu aplicación
    indices_a_excluir = list(range(36, 48))  # Índices de los puntos de los ojos
    puntos_referencia = np.delete(puntos_referencia, indices_a_excluir, axis=0)

    return puntos_referencia

def comparar_caras(imagen_entrada, imagen_referencia, nombre_referencia):
    puntos_referencia_entrada = extraer_puntos_referencia(imagen_entrada)
    puntos_referencia_referencia = extraer_puntos_referencia(imagen_referencia)

    # Calcula la diferencia entre los conjuntos de puntos de referencia.
    diferencia = np.linalg.norm(puntos_referencia_entrada - puntos_referencia_referencia)

    # Define un umbral para determinar si las caras son similares o no.
    umbral = 10.0

    # Calcula el porcentaje de coincidencia
    porcentaje_coincidencia = max(0, (umbral - diferencia) / umbral) * 100

    # Dibuja los puntos faciales y las líneas que los conectan en ambas imágenes
    imagen_entrada_puntos = imagen_entrada.copy()
    imagen_referencia_puntos = imagen_referencia.copy()

    for punto in puntos_referencia_entrada:
        cv2.circle(imagen_entrada_puntos, tuple(punto), 3, (0, 255, 0), -1)

    for punto in puntos_referencia_referencia:
        cv2.circle(imagen_referencia_puntos, tuple(punto), 3, (0, 255, 0), -1)

    for i in range(len(puntos_referencia_entrada)):
        cv2.line(imagen_entrada_puntos, tuple(puntos_referencia_entrada[i]), tuple(puntos_referencia_referencia[i]), (255, 0, 0), 1)

    # Guarda las imágenes con puntos en la carpeta "Imagenes"
    ruta_imagen_entrada_puntos = os.path.join('static/Imagenes', f'{nombre_referencia}_puntos.jpg')
    ruta_imagen_referencia_puntos = os.path.join('static/Imagenes', f'{nombre_referencia}_referencia_puntos.jpg')

    cv2.imwrite(ruta_imagen_entrada_puntos, imagen_entrada_puntos)
    cv2.imwrite(ruta_imagen_referencia_puntos, imagen_referencia_puntos)

    return diferencia, porcentaje_coincidencia, ruta_imagen_entrada_puntos, ruta_imagen_referencia_puntos

@app.route('/')
def index():
    return render_template('index.html', url_base=request.url_root)
@app.route('/comparar', methods=['POST'])
def comparar():
    try:
        # Obtiene la imagen de entrada del formulario
        imagen_entrada = request.files['imagen']
        nombre_imagen = imagen_entrada.filename  # Obtén el nombre original de la imagen
        
        # Convierte la imagen de entrada en un array numpy
        imagen_entrada_np = np.asarray(bytearray(imagen_entrada.read()), dtype=np.uint8)

        # Utiliza OpenCV para decodificar la imagen en formato BGR
        imagen_entrada_cv = cv2.imdecode(imagen_entrada_np, cv2.IMREAD_COLOR)

        # Compara la imagen de entrada con todas las imágenes en la carpeta 'Imagenes'
        directorio_imagenes = 'static/Imagenes/'
        resultados = []

        # Declara la variable imagen_entrada_puntos fuera del bloque try
        imagen_entrada_puntos = None

        for imagen_nombre in os.listdir(directorio_imagenes):
            if imagen_nombre.lower().endswith(('.jpg', '.jpeg', '.png')) and '_puntos' not in imagen_nombre and '_referencia_puntos' not in imagen_nombre:
                ruta_imagen_referencia = os.path.join(directorio_imagenes, imagen_nombre)

                try:
                    # Utiliza Pillow para cargar la imagen y convertirla a formato OpenCV
                    imagen_referencia_pil = Image.open(ruta_imagen_referencia)
                    imagen_referencia_cv = cv2.cvtColor(np.array(imagen_referencia_pil), cv2.COLOR_RGB2BGR)

                    # Realiza la comparación y obtiene los resultados
                    diferencia, porcentaje_coincidencia, ruta_imagen_entrada_puntos, ruta_imagen_referencia_puntos = comparar_caras(imagen_entrada_cv, imagen_referencia_cv, nombre_imagen)

                    resultados.append({
                        'imagen_referencia': url_for('static', filename=f'Imagenes/{imagen_nombre}'),
                        'imagen_entrada_puntos': url_for('static', filename=f'Imagenes/{imagen_nombre.replace(".", "_puntos.")}'),
                        'imagen_referencia_puntos': url_for('static', filename=f'Imagenes/{imagen_nombre.replace(".", "_referencia_puntos.")}'),
                        'diferencia': diferencia,
                        'porcentaje_coincidencia': porcentaje_coincidencia
                    })

                    # Asigna la variable imagen_entrada_puntos después de la comparación
                    imagen_entrada_puntos = ruta_imagen_entrada_puntos
                        
                except Exception as e:
                    # Añade un mensaje de registro para mostrar errores específicos al cargar imágenes
                    print(f"Error al cargar {ruta_imagen_referencia}: {str(e)}")

        # Ordena los resultados por porcentaje de similitud de mayor a menor
        resultados_ordenados = sorted(resultados, key=lambda x: x['porcentaje_coincidencia'], reverse=True)

        # Devuelve el resultado en formato JSON
        return jsonify(resultados_ordenados)
    except Exception as e:
        # Registra el error en la consola del servidor
        print(f"Error en la aplicación Flask: {str(e)}")
        # Devuelve un error 500 con el mensaje de error específico
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True)
