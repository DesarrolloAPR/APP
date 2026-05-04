import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 1. Configuración inicial
st.set_page_config(page_title="IA Prevención de Riesgos", layout="centered")

st.title("🛡️ Sistema de Control de EPP y Zonas")
st.write("Análisis de seguridad: Detección de casco y zona restringida.")

# Cargar el modelo
@st.cache_resource
def load_model():
    # IMPORTANTE: Si no tienes un modelo de cascos, yolov8n solo detectará personas.
    return YOLO('yolov8n.pt') 

model = load_model()

# 2. Configuración en la barra lateral
st.sidebar.header("Configuración de Seguridad")
conf_threshold = st.sidebar.slider("Confianza de la IA", 0.0, 1.0, 0.4)
detectar_zona = st.sidebar.checkbox("Activar zona restringida", value=True)

# 3. Entrada de cámara
img_file_buffer = st.camera_input("Toma una foto para inspección")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    # Ejecutar detección
    results = model(img_array, conf=conf_threshold)
    
    tiene_casco = False
    alerta_zona = False
    
    # Procesar resultados
    for r in results:
        # Dibujamos sobre una copia para no alterar el original
        annotated_frame = r.plot()
        h, w, _ = annotated_frame.shape
        
        for box in r.boxes:
            # Obtener nombre de clase
            cls_id = int(box.cls)
            label = model.names[cls_id].lower()
            
            # Lógica de Casco
            if "helmet" in label or "casco" in label:
                tiene_casco = True


        # Dibujar línea de zona (Color BGR en OpenCV)
        if detectar_zona:
            cv2.line(annotated_frame, (0, h//2), (w, h//2), (0, 0, 255), 3)

    # 4. Mostrar resultados
    st.image(annotated_frame, channels="RGB", caption="Resultado del Análisis")
    
    col1, col2 = st.columns(2)
    with col1:
        if tiene_casco:
            st.success("✅ Casco detectado")
        else:
            # Nota: Si usas yolov8n.pt, siempre entrará aquí porque no conoce 'cascos'
            st.warning("⚠️ Sin casco detectado")
            
    with col2:
        if alerta_zona:
            st.error("🚨 Invasión de zona")
        else:
            st.success("✅ Zona segura")

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# =========================================================
# 1. CONFIGURACIÓN DE RUTAS Y MODELO
# =========================================================
# Ruta donde YOLO guarda los resultados por defecto en Colab
PATH_ENTRENADO = 'runs/detect/train/weights/best.pt'
MODELO_BASE = 'yolov8n.pt'

@st.cache_resource
def load_model():
    # Si ya entrenaste, cargamos 'best.pt', si no, el base para que no de error
    if os.path.exists(PATH_ENTRENADO):
        st.sidebar.success("✅ Usando modelo entrenado (Cascos)")
        return YOLO(PATH_ENTRENADO)
    else:
        st.sidebar.info("ℹ️ Usando modelo base (Solo personas)")
        return YOLO(MODELO_BASE)

# =========================================================
# 2. FUNCIÓN DE ENTRENAMIENTO (Para ejecutar en Colab)
# =========================================================
def ejecutar_entrenamiento():
    st.info("Iniciando entrenamiento de Ultralytics...")
    model = YOLO(MODELO_BASE)
    # data.yaml debe estar en tu repo de GitHub
    model.train(data='data.yaml', epochs=50, imgsz=640, plots=True)
    st.success("¡Entrenamiento completado!")

# =========================================================
# 3. INTERFAZ STREAMLIT
# =========================================================
st.set_page_config(page_title="IA Prevención de Riesgos", layout="centered")
st.title("🛡️ Sistema de Control de EPP y Zonas")

model = load_model()

# Barra lateral
st.sidebar.header("Panel de Control")
if st.sidebar.button("🚀 Iniciar Entrenamiento ahora"):
    ejecutar_entrenamiento()
