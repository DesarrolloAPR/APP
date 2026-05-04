import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Configuración inicial
st.set_page_config(page_title="IA Prevención de Riesgos", layout="centered")

st.title("🛡️ Sistema de Control de EPP y Zonas")
st.write("Análisis de seguridad: Detección de casco y zona restringida.")

# 1. Cargar el modelo
@st.cache_resource
def load_model():
    # Nota: yolov8n detecta personas. Para cascos específicos se suele usar un modelo entrenado.
    return YOLO('yolov8n.pt') 

model = load_model()

# 2. Configuración en la barra lateral
st.sidebar.header("Configuración de Seguridad")
conf_threshold = st.sidebar.slider("Confianza de la IA", 0.0, 1.0, 0.5)
detectar_zona = st.sidebar.checkbox("Activar zona restringida", value=True)

# 3. Entrada de cámara
img_file_buffer = st.camera_input("Toma una foto para inspección")

if img_file_buffer is not None:
    # Preparar imagen
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    # Ejecutar detección
    results = model(img_array, conf=conf_threshold)
    
    # Variables de control
    tiene_casco = False
    alerta_zona = False
    
    for r in results:
        annotated_frame = r.plot()
        h, w, _ = annotated_frame.shape
        
        # Analizar cada objeto detectado
        for box in r.boxes:
            # Lógica de Casco (Basada en tu segundo pronto)
            # Filtramos por nombre de clase (asumiendo que el modelo tiene 'helmet' o 'casco')
            cls_id = int(box.cls)
            label = model.names[cls_id].lower()
            
            if "helmet" in label or "casco" in label:
                tiene_casco = True
            
            # Lógica de Zona (Basada en tu primer pronto)
            if detectar_zona:
                y1 = box.xyxy # Coordenada Y superior de la caja
                if y1 < h // 2:
                    alerta_zona = True

        # Dibujar línea de zona si está activa
        if detectar_zona:
            cv2.line(annotated_frame, (0, h//2), (w, h//2), (255, 0, 0), 5)

    # 4. Mostrar resultados y Alertas
    st.image(annotated_frame, caption="Resultado del Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if tiene_casco:
            st.success("✅ Casco detectado")
        else:
            st.error("❌ Sin casco visible")
            
    with col2:
        if alerta_zona:
            st.error("⚠️ Invasión de zona")
        else:
            st.success("✅ Zona segura")

    # Resumen final para consola/logs (tu lógica de print)
    print(f"Estado: Casco={tiene_casco}, Invasión={alerta_zona}")
