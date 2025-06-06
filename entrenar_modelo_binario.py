import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Rutas
ruta_dataset = "spectrograms/"

# Parámetros
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 8
EPOCHS = 20

# 1. Preparamos los datos
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    ruta_dataset,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # ← Cambiado a 'binary'
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    ruta_dataset,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # ← Cambiado a 'binary'
    subset='validation'
)

# 2. Construimos la CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # ← Cambiado a una sola salida binaria
])

# 3. Compilamos el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # ← Cambiado a 'binary_crossentropy'
              metrics=['accuracy'])

# 4. Entrenamos
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 5. Guardamos el modelo entrenado
os.makedirs('modelo_entrenado', exist_ok=True)
model.save('modelo_entrenado/modelo_llanto_bebe.h5')

print("✅ Modelo entrenado y guardado en 'modelo_entrenado/'")
