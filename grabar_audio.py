import sounddevice as sd
from scipy.io.wavfile import write
import os

def grabar_audio(nombre_archivo, duracion=5, fs=44100):
    print("Grabando...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    os.makedirs('data/grabaciones_crudas', exist_ok=True)
    write(f"data/grabaciones_crudas/{nombre_archivo}.wav", fs, audio)
    print("Grabación guardada.")

if __name__ == "__main__":
    nombre = input("Nombre del archivo (sin extensión): ")
    grabar_audio(nombre)