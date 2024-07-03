import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

# Cargar el archivo CSV
file_path = 'contactos2024_limpio.csv'
# Cargar los datos
data = pd.read_csv(file_path, delimiter=';')

print(data.head())  # Mostrar las primeras filas del archivo CSV para verificar la carga correcta

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

# Función para clasificar el mensaje y motivo
def classify_message(message, motivo):
    combined_text = f"{motivo} {message}".lower()

    if any(keyword in combined_text for keyword in ["taxímetro alterado", "cobro inadecuado", "cobro excesivo", "cobro", "taxímetro", "costo", "pagar", "pago"]):
        return 'problemas con el pago'
    elif any(keyword in combined_text for keyword in ["no contesta", "canceló solicitud", "ubicacion", "otro lado", "nunca llegó", "tiempo de espera largo", "no llegó a tiempo", "mal servicio", "vehículo incorrecto", "la unidad tardó demasiado tiempo", "llevó otro cliente", "la unidad no llegó", "vehículo desordenado", "no sale", "cancelo", "pesimo", "no desea la carrera", "no responde", "ya se", "no espera", "se ha ido", "seriedad", "carrera", "cancela", "esperando", "salió", "pide", "se fue", "doble", "sucio", "velocidad"]):
        return 'problemas con el viaje'
    elif any(keyword in combined_text for keyword in ["pre-registro", "#quieroserconductor", "#estoydispuestoalcambio"]):
        return 'mensaje de operador'
    elif any(keyword in combined_text for keyword in ["conducción peligrosa", "conducción brusca", "velocidad excesiva", "mal estado de vehículo", "teléfono hablo/chat", "etílico", "borracho", "mareado", "acompañante", "acompañado", "incorrecto", "peleando", "peliando", "loco"]):
        return 'problemas de seguridad'
    elif any(keyword in combined_text for keyword in ["app", "eliminar cuenta", "no me funciona de manera correcta", "verificación", "ubicación", "dirección"]):
        return 'problemas con la cuenta/app'
    elif any(keyword in combined_text for keyword in ["olvidó", "olvidado", "perdió", "olvidé", "se me cayó", "recuperar", "pérdida", "extravio", "olvido", "cayeron", "cayo", "perdida", "se me quedó", "se me quedo"]):
        return 'objetos perdidos'
    elif any(keyword in combined_text for keyword in ["desagradable", "comportamiento", "servicio poco amable", "actitud", "mal cliente", "acoso sexual", "agresivo", "exigente", "majadero", "prepotente", "grosero", "juega", "malcriado", "sexo", "grosero", "prepotente"]):
        return 'problemas de comportamiento'
    elif any(keyword in combined_text for keyword in ["hdp", "mmv", "vrg", "hp", "crvrg", "puta", "mrd", "verga", "hijo", "perra", "malparido", "pendejo", "caraverga", "gil"]):
        return 'vocabulario inadecuado'
    elif any(keyword in combined_text for keyword in ["bien", "excelente", "buen"]):
        return 'otros problemas'
    else:
        return 'otros problemas'

# Convertir todos los textos a minúsculas y clasificar los mensajes y motivos
data['mensaje'] = data['mensaje'].str.lower()
data['motivo'] = data['motivo'].str.lower()
data['categoria'] = data.apply(lambda row: classify_message(row['mensaje'], row['motivo']), axis=1)

# Preparar los datos
X = data.apply(lambda row: f"{row['motivo']} {row['mensaje']}", axis=1).astype(str).values
y = data['categoria'].values

# Codificar las etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizar y secuenciar el texto
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Construir y entrenar el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# Guardar el Tokenizer
tokenizer_path = 'tokenizer.pkl'
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Guardar las clases del LabelEncoder
np.save('classes.npy', label_encoder.classes_)

# Guardar el modelo
model.save('modelo_ticketing.keras')