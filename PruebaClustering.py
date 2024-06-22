import pandas as pd
import re
import tldextract
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Intentar leer el archivo CSV con opciones adicionales
try:
    df = pd.read_csv('contactos2024_final_limpio.csv', delimiter=';', quotechar='"', on_bad_lines='skip')
    print("Archivo CSV leído correctamente.")
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    df = None

# Continuar solo si el archivo se leyó correctamente
if df is not None:
    # Verificar si la columna 'mensaje' existe en el DataFrame
    if 'mensaje' not in df.columns:
        print("La columna 'mensaje' no se encuentra en el archivo CSV.")
    else:
        # Asegurarse de que todos los valores en 'mensaje' son cadenas de texto
        df['mensaje'] = df['mensaje'].astype(str)


        # 2. Preprocesamiento del texto
        def preprocess_text(text):
            # Convertir a minúsculas
            text = text.lower()

            # Eliminar URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

            # Eliminar direcciones de correo electrónico
            text = re.sub(r'\S+@\S+', '', text)

            # Eliminar dominios
            ext = tldextract.extract(text)
            text = text.replace(ext.domain + '.' + ext.suffix, '')

            # Eliminar caracteres especiales y números
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Eliminar espacios adicionales
            text = re.sub(r'\s+', ' ', text).strip()

            # Eliminar stop words
            stop_words = set(get_stop_words('es'))  # Asumiendo que el texto está en español
            text = ' '.join([word for word in text.split() if word not in stop_words])

            return text


        df['mensaje'] = df['mensaje'].apply(preprocess_text)

        # 3. Vectorización del texto
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df['mensaje'])

        # 4. Aplicación de clustering
        # Usando KMeans con un número arbitrario de clusters, digamos 10
        num_clusters = 20
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # Añadir las etiquetas de cluster al dataframe original
        df['cluster'] = kmeans.labels_

        # Guardar el dataframe con los clusters
        df.to_csv('contactos2024_final_limpio_clusterizado20.csv', index=False)

        # Imprimir los primeros 5 registros para verificar
        print(df.head())

