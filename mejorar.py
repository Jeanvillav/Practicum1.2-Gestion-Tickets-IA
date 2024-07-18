import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo, tokenizer y label encoder
model = tf.keras.models.load_model('modelo_ticketing.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
label_classes = np.load('classes.npy', allow_pickle=True)

# Cargar el dataset etiquetado manualmente
manual_file_path = 'etiquetadoManual.csv'
manual_data = pd.read_csv(manual_file_path)

# Asegurarse de que los textos están en minúsculas
manual_data['mensaje_completo'] = manual_data['mensaje_completo'].str.lower()

# Preparar los datos
X_manual = manual_data['mensaje_completo'].astype(str).values
y_manual = manual_data['categoria_predicha'].values

# Codificar las etiquetas
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes
y_manual = label_encoder.transform(y_manual)

# Tokenizar y secuenciar el texto
X_manual_seq = tokenizer.texts_to_sequences(X_manual)
max_length = 100  # Asegúrate de que sea el mismo valor utilizado previamente
X_manual_pad = pad_sequences(X_manual_seq, maxlen=max_length)

# Fine-tuning del modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_manual_pad, y_manual, epochs=5, batch_size=32, validation_split=0.2)

# Guardar el modelo ajustado
model.save('modelo_ticketing_ajustado.keras')