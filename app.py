import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="IA Prevención de Riesgos", layout="centered")

st.title("🛡️ Sistema de Control de EPP y Zonas")
st.write("Usa la cámara de tu móvil para detectar actos inseguros.")

# 1. Cargar el modelo (se descarga automáticamente la primera vez)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') # Modelo base: detecta personas

model = load_model()

# 2. Configuración en la barra lateral (Aquí puedes editar tú mismo)
st.sidebar.header("Configuración de Seguridad")
conf_threshold = st.sidebar.slider("Confianza de la IA", 0.0, 1.0, 0.5)
detectar_zona = st.sidebar.checkbox("Activar zona restringida", value=True)

# 3. Selector de cámara (Interfaz móvil)
img_file_buffer = st.camera_input("Toma una foto para inspeccionar el área")

if img_file_buffer is not None:
    # Convertir la imagen para que la IA la entienda
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    # Ejecutar detección
    results = model(img_array, conf=conf_threshold)
    
    # Dibujar resultados
    for r in results:
        annotated_frame = r.plot()
        
        # Lógica de zona (Ejemplo: Alerta si hay alguien en la mitad superior)
        if detectar_zona:
            height, width, _ = annotated_frame.shape
            # Dibujamos una línea roja de "Peligro"
            cv2.line(annotated_frame, (0, height//2), (width, height//2), (255, 0, 0), 5)
            
            for box in r.boxes:
                y1 = box.xyxy
                if y1 < height//2: # Si la cabeza está arriba de la línea
                    st.error("⚠️ ¡ALERTA! Trabajador en zona no permitida.")

    # Mostrar la imagen procesada
    st.image(annotated_frame, caption="Análisis en Tiempo Real")
# Asumiendo que 'foto' es la imagen capturada por tu cámara
deteccion = model(foto)
tiene_casco = len(deteccion.boxes) > 0

print("¿Lleva casco?:", "Sí" if tiene_casco else "No")
