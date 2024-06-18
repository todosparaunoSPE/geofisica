# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:13:01 2024

@author: jperezr
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io

# Título de la aplicación
st.title('Simulación Geofísica')
st.header("Creado por: Javier Horacio Pérez Ricárdez")

# Sección de Ayuda
def mostrar_ayuda():
    st.sidebar.subheader('Ayuda')
    st.sidebar.markdown("""
    ## Ayuda de la Aplicación

    1. **Importaciones y configuración inicial**:
       Se importan las librerías necesarias para la aplicación: Streamlit para la interfaz web, NumPy y Pandas para el manejo de datos, Matplotlib para las gráficas, y `io` para manejar la descarga de archivos.

    2. **Título de la aplicación**:
       Se establece el título de la aplicación, que en este caso es "Simulación Geofísica".

    3. **Configuración de parámetros de entrada en la barra lateral**:
       Se crea un control deslizante en la barra lateral que permite al usuario seleccionar la profundidad del pozo en metros.

    4. **Definición de modelos de velocidad de onda sísmica**:
       Se definen varios modelos de velocidad de onda sísmica que incluyen modelos constantes, lineales, exponenciales y parabólicos, así como una opción para un modelo personalizado.

    5. **Selección de modelo de velocidad**:
       Se permite al usuario seleccionar uno o más modelos de velocidad de onda sísmica desde la barra lateral.

    6. **Carga de datos personalizados desde un archivo CSV**:
       Se ofrece la opción de cargar un archivo CSV con datos personalizados de profundidad y velocidad de onda sísmica. Si se carga un archivo válido, los datos se interpolan y se utilizan para crear un modelo personalizado.

    7. **Funciones para cálculos y visualizaciones**:
       Se definen funciones para calcular el tiempo de viaje de las ondas sísmicas, personalizar los gráficos, y mostrar los resultados en una tabla comparativa. También se incluye una función para mostrar estadísticas de los tiempos de viaje y otra para descargar los resultados como un archivo CSV.

    8. **Cálculos y visualización de resultados**:
       Se ejecutan los cálculos para los modelos seleccionados y se generan gráficos que muestran los perfiles de tiempo de viaje. Los resultados se muestran en una tabla comparativa y se calculan estadísticas como el tiempo medio, la desviación estándar, y los tiempos mínimo y máximo. Se ofrece la opción de descargar los resultados en formato CSV.

    9. **Visualización del modelo 3D de velocidad de onda sísmica (ejemplo)**:
       Se muestra un ejemplo de un modelo 3D de velocidad de onda sísmica utilizando datos generados para fines ilustrativos.

    10. **Descripción final y pie de página**:
        Se proporciona una descripción final de la simulación y se añade un pie de página que indica el autor de la aplicación.
    """)

mostrar_ayuda()

# Parámetros de entrada
st.sidebar.header('Parámetros de Simulación')
depth = st.sidebar.slider('Profundidad del pozo (m)', min_value=100, max_value=5000, value=2000, step=100)

# Definición de modelos de velocidad de onda sísmica
velocity_models = {
    'Constante': lambda depth: np.full_like(depth, 3000),
    'Lineal (Gradiente 10 m/s/m)': lambda depth: 2000 + 10 * depth,
    'Lineal (Gradiente 20 m/s/m)': lambda depth: 2000 + 20 * depth,
    'Exponencial': lambda depth: 2000 * np.exp(0.0001 * depth),
    'Parabólico': lambda depth: 2000 + 0.5 * depth**2,
    'Personalizado': None
}

# Selección de modelo de velocidad de onda sísmica
velocity_model_selected = st.sidebar.multiselect('Modelos de Velocidad de Onda Sísmica', list(velocity_models.keys()), default=['Constante'])

# Cargar datos personalizados
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV con datos personalizados", type="csv")
if uploaded_file is not None:
    custom_data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:", custom_data)
    if 'depth' in custom_data.columns and 'velocity' in custom_data.columns:
        velocity_models['Personalizado'] = lambda depth: np.interp(depth, custom_data['depth'], custom_data['velocity'])

# Función para calcular el tiempo de viaje de la onda sísmica
def calculate_travel_time(depth, velocity):
    return 2 * depth / velocity

# Parámetros de personalización del gráfico
st.sidebar.header('Personalización del Gráfico')
color = st.sidebar.color_picker('Selecciona el color de la línea', '#00f900')
line_style = st.sidebar.selectbox('Estilo de línea', ['-', '--', '-.', ':'])

# Función para graficar el perfil de tiempo de viaje
def plot_travel_time(depth, velocity, label=None):
    depths = np.linspace(0, depth, 100)
    times = calculate_travel_time(depths, velocity)
    plt.plot(times, depths, label=label, color=color, linestyle=line_style)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Profundidad (m)')
    plt.title('Perfil de Tiempo de Viaje de Onda Sísmica')

# Función para mostrar los datos de tiempo de viaje en una tabla comparativa
def show_comparison_table(models_data):
    model_names = []
    depth_values = []
    time_values = []

    for model_data in models_data:
        model_name = model_data['Modelo']
        depths = model_data['Profundidad (m)']
        times = model_data['Tiempo de Viaje (s)']
        model_names.extend([model_name] * len(depths))
        depth_values.extend(depths)
        time_values.extend(times)

    df = pd.DataFrame({
        'Modelo': model_names,
        'Profundidad (m)': depth_values,
        'Tiempo de Viaje (s)': time_values
    })

    st.subheader('Tabla Comparativa de Tiempos de Viaje de Onda Sísmica')
    st.dataframe(df)

# Función para mostrar estadísticas de los tiempos de viaje
def show_statistics(models_data):
    stats = []
    for model_data in models_data:
        model_name = model_data['Modelo']
        times = model_data['Tiempo de Viaje (s)']
        stats.append({
            'Modelo': model_name,
            'Tiempo Medio (s)': np.mean(times),
            'Desviación Estándar (s)': np.std(times),
            'Tiempo Mínimo (s)': np.min(times),
            'Tiempo Máximo (s)': np.max(times)
        })

    df_stats = pd.DataFrame(stats)
    st.subheader('Estadísticas de Tiempos de Viaje')
    st.dataframe(df_stats)

# Función para descargar resultados
def download_results(models_data):
    output = io.StringIO()
    df = pd.DataFrame(models_data)
    df.to_csv(output, index=False)
    return output.getvalue()

# Cálculos y resultados
st.subheader('Resultados')

if not velocity_model_selected:
    st.warning('Por favor selecciona al menos un modelo de velocidad de onda sísmica.')
else:
    models_data = []
    for model_name in velocity_model_selected:
        if model_name == 'Personalizado':
            velocity_custom = st.sidebar.number_input('Velocidad de onda sísmica personalizada (m/s)', min_value=1000, max_value=6000, value=3000, step=100)
            if velocity_custom is not None:
                velocity = velocity_custom
            else:
                velocity = None
        else:
            velocity = velocity_models[model_name](depth)

        if velocity is not None:
            time = calculate_travel_time(depth, velocity)
            models_data.append({
                'Modelo': model_name,
                'Profundidad (m)': np.linspace(0, depth, 100),
                'Tiempo de Viaje (s)': calculate_travel_time(np.linspace(0, depth, 100), velocity)
            })

            plot_travel_time(depth, velocity, label=model_name)

    plt.legend()
    st.pyplot()

    if models_data:
        show_comparison_table(models_data)
        show_statistics(models_data)

        if st.button('Descargar Resultados'):
            csv = download_results(models_data)
            st.download_button(
                label="Descargar como CSV",
                data=csv,
                file_name='resultados_simulacion.csv',
                mime='text/csv',
            )

# Visualización del modelo 3D de velocidad de onda sísmica (ejemplo)
st.subheader('Modelo 3D de Velocidad de Onda Sísmica (ejemplo)')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Datos de ejemplo para modelo 3D
x = np.linspace(0, 100, 50)
y = np.linspace(0, 100, 50)
x, y = np.meshgrid(x, y)
z = x**2 + y**2

ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Velocidad de Onda Sísmica')
ax.set_title('Modelo 3D de Velocidad de Onda Sísmica')

st.pyplot(fig)

st.markdown("""
En esta simulación, puedes seleccionar varios modelos de velocidad de onda sísmica y comparar sus perfiles de tiempo de viaje lado a lado en una tabla comparativa organizada y fácil de entender. Además, se muestra un ejemplo de modelo 3D de velocidad de onda sísmica para fines ilustrativos.
""")

#st.markdown("---")
#st.markdown("Creado por: Javier Horacio Pérez Ricárdez")