import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar el archivo CSV con los nuevos datos
new_data_path = 'contactos2024_limpio.csv'

try:
    # Intentamos leer el archivo CSV con ';' como delimitador
    new_data = pd.read_csv(new_data_path, delimiter=';')
except Exception as e:
    print(f"Error leyendo el archivo CSV: {e}")
    exit()

# Verificar si las columnas necesarias existen
required_columns = ['motivo', 'mensaje']
for col in required_columns:
    if col not in new_data.columns:
        print(f"El archivo CSV no contiene la columna '{col}'. Verifica el formato del archivo.")
        exit()

# Combinar 'mensaje' y 'motivo' en una sola columna para el preprocesamiento
new_data['mensaje_completo'] = new_data['motivo'].astype(str) + ' ' + new_data['mensaje'].astype(str)

# Cargar el modelo previamente entrenado
model_path = 'modelo_ticketing.keras'
model = tf.keras.models.load_model(model_path)

# Cargar el Tokenizer utilizado durante el entrenamiento
tokenizer_path = 'tokenizer.pkl'
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Preprocesar los nuevos datos
X_new = new_data['mensaje_completo'].astype(str).values
X_new_seq = tokenizer.texts_to_sequences(X_new)
max_length = 100
X_new_pad = pad_sequences(X_new_seq, maxlen=max_length)

# Clasificar los nuevos mensajes
predictions = model.predict(X_new_pad)

# Decodificar las predicciones
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
predicted_categories = label_encoder.inverse_transform([np.argmax(pred) for pred in predictions])

# Asignar prioridades a las categorías predichas
category_priority = {
    'Problemas con el pago': 1,
    'Problemas con el viaje': 2,
    'Problemas de seguridad': 1,
    'Problemas con la cuenta/app': 2,
    'Problemas de comportamiento': 2,
    'Mensaje de operador': 3,
    'Otros problemas': 3
}

priorities = [category_priority.get(category, 3) for category in predicted_categories]

# Agregar las categorías predichas y sus prioridades al DataFrame original
new_data['categoria_predicha'] = predicted_categories
new_data['prioridad_predicha'] = priorities

# Guardar los resultados en un nuevo archivo CSV
output_file_path = 'resultados_prediccion.csv'
new_data.to_csv(output_file_path, index=False)
