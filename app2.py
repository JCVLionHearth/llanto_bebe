import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy.io.wavfile import write
import tempfile
from PIL import Image

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('modelo_entrenado/modelo_llanto_bebe.h5')

# Etiquetas de las clases
clases = ['Cansancio', 'Dolor', 'Hambre', 'Incomodidad']

# Función para grabar audio
def grabar_audio(duracion=5, fs=44100):
    st.info(f"Grabando {duracion} segundos...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio, fs

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

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_img.name, bbox_inches='tight', pad_inches=0)
    plt.close()
    return temp_img.name

# Función para predecir a partir del espectrograma
def predecir(espectrograma_path):
    img = Image.open(espectrograma_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array)
    idx = np.argmax(prediccion)
    clase_predicha = clases[idx]
    confianza = float(prediccion[0][idx])
    return clase_predicha, confianza

# Streamlit UI
st.title("Clasificador de Llanto de Bebé")
st.write("Este modelo predice la causa del llanto: Hambre, Dolor, Incomodidad o Cansancio.")

opcion = st.selectbox("¿Cómo deseas ingresar el audio?", ["Grabar ahora", "Subir archivo .wav"])

if opcion == "Grabar ahora":
    if st.button("Grabar Audio"):
        audio, fs = grabar_audio()
        espectrograma_path = crear_espectrograma(audio, fs)
        st.image(espectrograma_path, caption="Espectrograma generado", use_column_width=True)
        clase, confianza = predecir(espectrograma_path)
        st.success(f"Predicción: **{clase}** (Confianza: {confianza:.2f})")

elif opcion == "Subir archivo .wav":
    archivo_audio = st.file_uploader("Sube un archivo de audio (.wav)", type=["wav"])
    if archivo_audio is not None:
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path.write(archivo_audio.read())

        y, sr = librosa.load(temp_audio_path.name, sr=None)
        audio = np.expand_dims(y, axis=1)

        espectrograma_path = crear_espectrograma(audio, sr)
        st.image(espectrograma_path, caption="Espectrograma generado", use_column_width=True)
        clase, confianza = predecir(espectrograma_path)
        st.success(f"Predicción: **{clase}** (Confianza: {confianza:.2f})")
