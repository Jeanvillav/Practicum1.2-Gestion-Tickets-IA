import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('contactos2024.csv', delimiter=';', na_filter=False)

# Eliminar columnas no deseadas
df = df.drop(['hashTag', 'longitud', 'comentario', 'latitud'], axis=1)

# Limpiar columna mensaje: quitar los espacios y el guion (-) al principio
df['mensaje'] = df['mensaje'].str.replace('-', '').str.strip()

# Crear una nueva columna para la calificación con valor inicial 0
df['calificación'] = 0

# Asignar valores a la columna calificación basado en el contenido de la columna mensaje
df.loc[df['mensaje'].str.contains("Calificación 1 estrella<br>"), 'calificación'] = 1
df.loc[df['mensaje'].str.contains("Calificación 2 estrellas<br>"), 'calificación'] = 2

# Eliminar substrings "Calificación 1 estrella<br>" y "Calificación 2 estrellas<br>" de la columna mensaje
df['mensaje'] = df['mensaje'].str.replace("Calificación 1 estrella<br>", "").str.replace("Calificación 2 estrellas<br>", "")

# Eliminar filas con valores nulos
df = df.dropna(how='any')
df = df.dropna(subset=['ciudad'])
# Convertir todas las filas de todas las columnas a minúsculas
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df['mensaje'] = df['mensaje'].str.lstrip()
# Mostrar el dataset procesado
print(df)

# Guardar el dataset procesado en un nuevo archivo CSV si es necesario
df.to_csv('contactos2024_limpio.csv', index=False, sep=';')
