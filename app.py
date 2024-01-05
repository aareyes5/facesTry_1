from flask import Flask, render_template, request, g
import pyodbc  # Librería para conectar con SQL Server
import face_recognition
import numpy as np
from PIL import Image
import io
import base64 

app = Flask(__name__)

# Configuración de la base de datos SQL Server
DATABASE_CONFIG = {
    'server': 'MASTER',
    'database': 'CARAS',
    'username': 'FACE',
    'password': 'FACE',
    'driver': '{ODBC Driver 17 for SQL Server}',  # Asegúrate de tener el controlador correcto instalado
}
# Antes de cada solicitud, establecer la conexión a la base de datos
@app.before_request
def antes_de_la_solicitud():
    g.bd = conectar_bd()

# Después de cada solicitud, cerrar la conexión a la base de datos
@app.teardown_request
def despues_de_la_solicitud(excepcion):
    if hasattr(g, 'bd'):
        g.bd.close()

def conectar_bd():
    return pyodbc.connect(
        f"DRIVER={DATABASE_CONFIG['driver']};SERVER={DATABASE_CONFIG['server']};DATABASE={DATABASE_CONFIG['database']};"
        f"Trusted_Connection=yes;"  # Autenticación de Windows
    )

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la captura de imágenes y procesar los resultados
@app.route('/upload', methods=['POST'])
def upload():
    # Obtener la imagen desde el formulario
    imagen_codificada = request.form['image']

    # Decodificar la imagen base64
    imagen_decodificada = Image.open(io.BytesIO(base64.b64decode(imagen_codificada.split(",")[1])))
    
    # Convertir la imagen de PIL a un array NumPy
    imagen_np = np.array(imagen_decodificada)

    # Encontrar caras en la imagen utilizando face_recognition
    caras_en_imagen = face_recognition.face_locations(imagen_np)

    # Si hay caras en la imagen
    if caras_en_imagen:
        # Aquí deberías implementar la lógica de análisis facial y cálculo de la calificación de depresión
        # Utiliza la información en 'caras_en_imagen' para obtener regiones específicas de la imagen que contienen caras

        # Ejemplo de inserción en la base de datos SQL Server
        with g.bd.cursor() as cursor:
            cursor.execute("INSERT INTO resultados (usuario_id, calificacion_depresion) VALUES (?, ?)", (1, 75))
            g.bd.commit()

        return render_template('result.html', depression_score=75)  # Reemplaza 75 con la calificación real obtenida

    else:
        return render_template('result.html', depression_score=0)  # Sin caras, calificación 0

if __name__ == '__main__':
    app.run(debug=True)