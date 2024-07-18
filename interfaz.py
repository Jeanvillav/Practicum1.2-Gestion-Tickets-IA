from flask import Flask, request, render_template
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ajusta la ruta al modelo
model_path = 'C:/Users/villa/OneDrive/Documents/Practicum1.2/modelo_ticketing_ajustado.keras'
model = tf.keras.models.load_model(model_path)

with open('C:/Users/villa/OneDrive/Documents/Practicum1.2/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

label_classes = np.load('C:/Users/villa/OneDrive/Documents/Practicum1.2/classes.npy', allow_pickle=True)

# Definir las prioridades para cada categoría
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

# Crear la aplicación Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        motivo = request.form['motivo']
        mensaje = request.form['mensaje']
        mensaje_completo = f"{motivo} {mensaje}".lower()
        
        # Tokenizar y secuenciar el mensaje
        seq = tokenizer.texts_to_sequences([mensaje_completo])
        pad_seq = pad_sequences(seq, maxlen=100)
        
        # Hacer la predicción
        pred = model.predict(pad_seq)
        categoria_predicha = label_classes[np.argmax(pred)]
        prioridad_predicha = category_priority.get(categoria_predicha, 'No asignada')
        
        return render_template('index.html', motivo=motivo, mensaje=mensaje, categoria=categoria_predicha, prioridad=prioridad_predicha)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
