import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def crear_espectrograma(ruta_audio, ruta_salida, nombre_imagen):
    """
    Función original para crear un espectrograma individual
    """
    y, sr = librosa.load(ruta_audio, sr=None)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    os.makedirs(ruta_salida, exist_ok=True)
    plt.savefig(f"{ruta_salida}/{nombre_imagen}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def procesar_carpeta_audios(carpeta_entrada, carpeta_salida):
    """
    Nueva función para procesar todos los audios de una carpeta
    usando la función crear_espectrograma original
    """
    # Verificar si la carpeta existe
    if not os.path.exists(carpeta_entrada):
        print(f"Error: La carpeta {carpeta_entrada} no existe")
        return

    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Procesar cada archivo de audio
    for archivo in os.listdir(carpeta_entrada):
        if archivo.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            try:
                # Construir rutas
                ruta_audio = os.path.join(carpeta_entrada, archivo)
                nombre_base = os.path.splitext(archivo)[0]
                nombre_imagen = f"{nombre_base}_spectrograma"
                
                # Usar tu función original
                crear_espectrograma(ruta_audio, carpeta_salida, nombre_imagen)
                print(f"✓ {archivo} -> {nombre_imagen}.png")
                
            except Exception as e:
                print(f"✗ Error procesando {archivo}: {str(e)}")

if __name__ == "__main__":
    # Configuración de rutas
    CARPETA_AUDIOS = "data\incomodidad"
    CARPETA_SALIDA = "spectrograms\incomodidad"
    
    # Procesar todos los audios
    print(f"\nIniciando procesamiento de audios en {CARPETA_AUDIOS}")
    procesar_carpeta_audios(CARPETA_AUDIOS, CARPETA_SALIDA)
    print("\nProceso completado. Espectrogramas guardados en:", CARPETA_SALIDA)