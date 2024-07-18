import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el modelo, tokenizer y label encoder
model = tf.keras.models.load_model('modelo_ticketing_ajustado.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

label_classes = np.load('classes.npy', allow_pickle=True)

# Cargar el dataset etiquetado manualmente
manual_file_path = 'etiquetadom.csv'
manual_data = pd.read_csv(manual_file_path)

# Asegurarse de que los textos están en minúsculas
manual_data['mensaje_completo'] = manual_data['mensaje_completo'].str.lower()

# Preparar los datos
X_manual = manual_data['mensaje_completo'].astype(str).values
y_manual = manual_data['categoria_predicha'].values

# Codificar las etiquetas
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes
y_manual_encoded = label_encoder.transform(y_manual)

# Tokenizar y secuenciar el texto
X_manual_seq = tokenizer.texts_to_sequences(X_manual)
max_length = 100  # Asegúrate de que sea el mismo valor utilizado previamente
X_manual_pad = pad_sequences(X_manual_seq, maxlen=max_length)

# Hacer predicciones
manual_predictions = model.predict(X_manual_pad)
manual_predicted_categories = [label_classes[np.argmax(pred)] for pred in manual_predictions]
manual_predicted_encoded = label_encoder.transform(manual_predicted_categories)

# Evaluar precisión
accuracy = accuracy_score(y_manual_encoded, manual_predicted_encoded)
print(f'Precisión: {accuracy}')

# Reporte de clasificación
report = classification_report(y_manual_encoded, manual_predicted_encoded, target_names=label_classes)
print(report)

# Matriz de confusión
conf_matrix = confusion_matrix(y_manual_encoded, manual_predicted_encoded)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_classes, yticklabels=label_classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
