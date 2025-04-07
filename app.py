# Paula Betina Reyes Anaya

# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Configuramos la página de Streamlit
st.set_page_config(page_title="Aplicación de detección de objetos", 
                   page_icon="happy.png", 
                   layout="centered",
                   initial_sidebar_state="auto")

# Ocultar elementos de Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Interfaz de usuario
with st.sidebar:
    st.image('Objetos.png', caption='Imagen de ejemplo de los objetos a detectar', use_container_width=True)
    st.title("Entrenamiento de un modelo YOLOv8 para detector de objetos personalizado")
    st.subheader("Detecta botas, guantes, cascos, chalecos y personas en imágenes")
    st.write("Capture una imagen para realizar la detección de los objetos mencionados.")

st.image('logo.png', use_container_width=True)
st.markdown('<h3 style="font-size: 18px;">Elaborado por: Paula Betina Reyes Anaya', unsafe_allow_html=True)

# Definimos el título y la descripción de la aplicación
st.title("Universidad Autónoma de Bucaramanga - UNAB")
st.header('Aplicación para detectar objetos con YOLOv8', divider='rainbow')
st.subheader("Entrenar un modelo YOLOv8 para un detector de objetos personalizado como botas, guantes, casco, humano y chaleco")

with st.container(border=True):
    st.subheader("Detección de Equipos de Protección Personal")
    st.write("Realizado por Paula Betina Reyes :wave:")
    st.write("""
**OBJETIVO**:
Esta aplicación permite al usuario capturar o cargar una imagen para detectar objetos específicos (botas, guantes, cascos, humanos y chalecos) mediante un modelo YOLOv8 entrenado específicamente para estas clases.
""")

with st.container(border=True):
    st.subheader("TRABAJO GOOGLE COLAB")
    st.write("""Enlace del modelo entrenado en Google Colab con las librerías:
    https://colab.research.google.com/drive/11pruICJyx5VFHeWBX_wklpTmNKDprzcX?usp=sharing""")

# Subtítulo visual
st.markdown("<h2 style='text-align: center;'>Sube una foto para detectar botas, guantes, cascos, chalecos y persona</h2>", unsafe_allow_html=True)

# Cargar el modelo YOLOv8 entrenado
model = YOLO("best.pt")
try:
    model = YOLO("best.pt")
    st.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error("❌ Error al cargar el modelo")
    st.exception(e)
    st.stop()


# Elegir método de entrada
option = st.radio("Selecciona el método de entrada:", ("📸 Cámara", "🖼️ Subir imagen"))

# Función para mostrar resultados de detección
def mostrar_resultado(imagen):
    results = model.predict(imagen, conf=0.25)
    pred = results[0].plot()
    st.image(pred, caption="Resultado de la detección", use_container_width=True)

# 📸 Entrada por cámara
if option == "📸 Cámara":
    img_file_buffer = st.camera_input("Capture una foto para identificar un producto")    
    if img_file_buffer is None:
        st.info("Por favor tome una foto")
    else:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Imagen capturada", use_column_width=True)
        mostrar_resultado(image)

# 🖼️ Subir imagen
elif option == "🖼️ Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        mostrar_resultado(image)
