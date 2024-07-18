import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Cargar el modelo, tokenizer y label encoder
model = tf.keras.models.load_model('modelo_ticketing_ajustado.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

label_classes = np.load('classes.npy', allow_pickle=True)

# Definir las categorías y sus prioridades
category_priority = {
    'problemas con el pago': 'Inmediata',
    'problemas con el viaje': 'Alta',
    'problemas de seguridad': 'Inmediata',
    'problemas con la cuenta/app': 'Alta',
    'problemas de comportamiento': 'Urgente',
    'mensaje de operador': 'Baja',
    'otros problemas': 'Baja',
    'objetos perdidos': 'Urgente',
    'vocabulario inadecuado': 'Alta'
}

# Cargar el archivo CSV
file_path = 'contactos2024_limpio.csv'
data = pd.read_csv(file_path, delimiter=';')

# Convertir todos los textos a minúsculas y combinar motivo y mensaje
data['mensaje_completo'] = data.apply(lambda row: f"{row['motivo']} {row['mensaje']}".lower(), axis=1)

# Tokenizar y secuenciar el texto
X_data_seq = tokenizer.texts_to_sequences(data['mensaje_completo'])
max_length = 100
X_data_pad = pad_sequences(X_data_seq, maxlen=max_length)

# Hacer predicciones
predictions = model.predict(X_data_pad)
predicted_categories = [label_classes[np.argmax(pred)] for pred in predictions]

# Agregar las predicciones al DataFrame
data['categoria_predicha'] = predicted_categories
data['prioridad_predicha'] = data['categoria_predicha'].map(category_priority)

# Guardar el archivo CSV con las predicciones
output_file_path = 'contactos2024_clasificado.csv'
data.to_csv(output_file_path, index=False, sep=';')

print(f'Archivo clasificado guardado en: {output_file_path}')
