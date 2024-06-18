import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Verificación del uso de GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Is the GPU being used:", tf.config.list_physical_devices('GPU'))

# Cargar el archivo CSV
file_path = 'contactos2024_limpio.csv'

# Intenta leer el archivo CSV con diferentes delimitadores
try:
    data = pd.read_csv(file_path, delimiter=',')
except pd.errors.ParserError:
    data = pd.read_csv(file_path, delimiter=';')

print(data.head())  # Mostrar las primeras filas del archivo CSV para verificar la carga correcta

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

# Función para clasificar el mensaje y motivo
def classify_message(message, motivo):
    combined_text = f"{motivo} {message}".lower()
    
    if any(keyword in combined_text for keyword in ["cobro", "quiere llevar sin taxímetro", "taxímetro", "cobró", "pagó", "costo de la carrera", "pagar", "pago", "cobrar", "justo"]):
        return 'Problemas con el pago'
    elif any(keyword in combined_text for keyword in ["nunca llegó", "no contestaba", "no contesta", "canceló solicitud", "nunca sali", "no atendió", "la unidad", "no quiso llevar", "no salió", "no respondió", "no sale", "no desea la carrera", "dirección indicado", "mala ubicación", "no responde", "tampoco contesta", "se fue", "otro taxi", "llevó otro cliente", "no espera", "hace pasar tiempo", "se ha ido", "dirección incorrecta", "se ido", "carrera no realizada", "otro lado", "atendió solicitud", "canceló solicitud", "finaliza la carrera"]):
        return 'Problemas con el viaje'
    elif any(keyword in combined_text for keyword in ["quiero ser conductor", "pre-registro", "quiero trabajar", "cómo ser conductor", "cómo trabajar", "trabajar en", "#estoydispuestoalcambio", "|", "#quieroserconductor"]):
        return 'Mensaje de operador'
    elif any(keyword in combined_text for keyword in ["inseguro", "seguridad", "inseguridad"]):
        return 'Problemas de seguridad'
    elif any(keyword in combined_text for keyword in ["contactar", "aplicativo", "eliminar cuenta", "código", "app", "registro", "cuenta", "problema con cuenta", "app no funciona"]):
        return 'Problemas con la cuenta/app'
    elif any(keyword in combined_text for keyword in ["asco de sujeto", "sexual", "uso teléfono hablo / chat", "actitud", "actitud desagradable", "desagradable", "mal cliente", "burla", "sexuales", "mal comportamiento"]):
        return 'Problemas de comportamiento'
    else:
        return 'Otros problemas'

# Clasificar los mensajes y motivos
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

# Desplegar el modelo para clasificar nuevos mensajes
def classify_new_message(message, motivo):
    combined_text = f"{motivo} {message}".lower()
    seq = tokenizer.texts_to_sequences([combined_text])
    pad = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(pad)
    category = label_encoder.inverse_transform([np.argmax(pred)])
    return category[0]

# Ejemplo de uso
new_message = "El conductor nunca llegó y no me contestaba los mensajes"
new_motivo = "solicitud cliente"
category = classify_new_message(new_message, new_motivo)
priority = category_priority.get(category, 3)
print(f'Categoría: {category}, Prioridad: {priority}')

# Guardar el modelo
model.save('modelo_ticketing.keras')
