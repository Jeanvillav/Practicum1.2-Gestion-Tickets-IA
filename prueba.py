import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Verificación del uso de GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Is the GPU being used:", tf.config.list_physical_devices('GPU'))

# Cargar el modelo previamente entrenado
model_path = 'modelo_ticketing.keras'
model = tf.keras.models.load_model(model_path)

# Cargar el Tokenizer utilizado durante el entrenamiento
tokenizer_path = 'tokenizer.pkl'
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Cargar las clases del LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Definir las categorías y sus prioridades
category_priority = {
    'Problemas con el pago': 1,
    'Problemas con el viaje': 2,
    'Problemas de seguridad': 1,
    'Problemas con la cuenta/app': 2,
    'Problemas de comportamiento': 2,
    'Mensaje de operador': 3,
    'Otros problemas': 3
}

# Función para clasificar un nuevo mensaje y motivo
def classify_new_message(motivo, mensaje):
    combined_text = f"{motivo} {mensaje}".lower()
    seq = tokenizer.texts_to_sequences([combined_text])
    pad = pad_sequences(seq, maxlen=100)
    pred = model.predict(pad)
    category = label_encoder.inverse_transform([np.argmax(pred)])
    priority = category_priority.get(category[0], 3)
    return category[0], priority

# Ejemplo de uso
new_message = "El conductor nunca llegó y no me contestaba los mensajes"
new_motivo = "solicitud cliente"
category, priority = classify_new_message(new_motivo, new_message)
print(f'Categoría: {category}, Prioridad: {priority}')

# Otro ejemplo
custom_message = input("Ingrese el mensaje: ")
custom_motivo = input("Ingrese el motivo: ")
custom_category, custom_priority = classify_new_message(custom_motivo, custom_message)
print(f'Categoría: {custom_category}, Prioridad: {custom_priority}')