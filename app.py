# app.py (versión Flask)
from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy.io.wavfile import write
import tempfile
from PIL import Image
import io
import base64

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('modelo_entrenado/modelo_llanto_bebe.h5')

# Etiquetas de las clases
clases = ['Cansancio', 'Dolor', 'Hambre', 'Incomodidad']

# Función para crear espectrograma desde el audio
def crear_espectrograma(audio, fs):
    audio_float = audio.flatten().astype(np.float32)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_wav.name, fs, audio_float)

    y, sr = librosa.load(temp_wav.name, sr=None)

    # Crear espectrograma
    fig, ax = plt.subplots(figsize=(3, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')

    # Guardar en buffer en lugar de archivo temporal
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    img_buffer.seek(0)
    
    return img_buffer

# Función para predecir
def predecir(espectrograma_buffer):
    img = Image.open(espectrograma_buffer).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array)
    idx = np.argmax(prediccion)
    clase_predicha = clases[idx]
    confianza = float(prediccion[0][idx])
    return clase_predicha, confianza

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    try:
        # Leer el archivo de audio
        y, sr = librosa.load(audio_file, sr=None)
        audio = np.expand_dims(y, axis=1)
        
        # Procesar
        espectrograma_buffer = crear_espectrograma(audio, sr)
        clase, confianza = predecir(espectrograma_buffer)
        
        # Convertir imagen a base64 para mostrarla
        espectrograma_buffer.seek(0)
        img_base64 = base64.b64encode(espectrograma_buffer.read()).decode('utf-8')
        
        return jsonify({
            'prediction': clase,
            'confidence': confianza,
            'image': img_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)