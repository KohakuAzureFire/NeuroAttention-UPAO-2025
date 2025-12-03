import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="NeuroAttention | UPAO",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CABECERA ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2814/2814666.png", width=100)
with col2:
    st.title("NeuroAttention Predictor")
    st.markdown("##### Sistema de Aprendizaje Estadístico (Random Forest Real)")
    st.caption("Proyecto UPAO 2025")

st.divider()

# --- CARGAR MODELO REAL (Silencioso) ---
@st.cache_resource
def cargar_modelo():
    try:
        # Busca dentro de la carpeta 'model/'
        model = joblib.load('model/modelo_random_forest.pkl')
        cols = joblib.load('model/columnas_modelo.pkl')
        return model, cols
    except Exception as e:
        return None, None

modelo, columnas_entrenamiento = cargar_modelo()

# --- SIDEBAR (ENTRADA DE DATOS) ---
with st.sidebar:
    st.header("📝 Perfil del Niño")
    # Se eliminó el mensaje de estado (Success/Error) para limpiar la interfaz
    
    with st.expander("Información Demográfica", expanded=True):
        edad = st.slider('Edad (años)', 1, 18, 10)
        genero = st.selectbox('Género', ['Masculino', 'Femenino', 'Otro'])
    
    with st.expander("Hábitos Digitales", expanded=True):
        horas_pantalla = st.slider('Horas diarias de pantalla', 0.0, 12.0, 3.5, step=0.5)
        # Opciones alineadas con tu dataset
        tipo_pantalla = st.selectbox('Contenido Principal', ['Educativo', 'Recreacional', 'Mixto'])
        tipo_dia = st.radio('Contexto', ['Día de semana', 'Fin de semana'])

# --- PROCESAMIENTO DE DATOS (CORREGIDO) ---
def procesar_entrada_real(edad, genero, horas, tipo_p, tipo_d, columnas_modelo):
    # 1. Crear plantilla con todas las columnas en 0
    data = {col: 0 for col in columnas_modelo}
    
    # 2. Asignar variables numéricas (CORRECCIÓN DE NOMBRE)
    # Tu modelo fue entrenado con "(hours)", así que usamos ese nombre exacto
    if 'Average Screen Time (hours)' in data:
        data['Average Screen Time (hours)'] = horas
    else:
        # Respaldo por si acaso cambiaste el nombre
        data['Average Screen Time'] = horas
        
    if 'Age' in data: data['Age'] = edad
    if 'Sample Size' in data: data['Sample Size'] = 120 
    
    # 3. Mapeo inteligente de variables categóricas (One-Hot)
    # GÉNERO
    if genero == 'Masculino' and 'Gender_Male' in data: 
        data['Gender_Male'] = 1
    elif genero == 'Otro':
        # Busca cualquier columna que diga 'Gender' y 'Other' (para evitar errores de texto exacto)
        for col in data:
            if 'Gender' in col and 'Other' in col:
                data[col] = 1
    # Nota: 'Femenino' no hace nada porque es la clase base (todos 0)

    # TIPO DE PANTALLA
    if tipo_p == 'Recreacional' and 'Screen Time Type_Recreational' in data: 
        data['Screen Time Type_Recreational'] = 1
    elif tipo_p == 'Mixto':
        # En tu dataset 'Mixto' suele ser 'Total' o similar
        if 'Screen Time Type_Total' in data: data['Screen Time Type_Total'] = 1
    # Nota: 'Educativo' es la clase base

    # TIPO DE DÍA
    if tipo_dia == 'Fin de semana' and 'Day Type_Weekend' in data: 
        data['Day Type_Weekend'] = 1
    # Nota: 'Día de semana' es la clase base
    
    return pd.DataFrame([data])

# --- INTERFAZ PRINCIPAL ---
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader("📊 Resultado del Análisis")
    
    if st.button('🚀 EJECUTAR PREDICCIÓN CON IA'):
        
        # Verificar si el modelo cargó bien
        if modelo is not None and columnas_entrenamiento is not None:
            try:
                # 1. Procesar datos
                df_real = procesar_entrada_real(edad, genero, horas_pantalla, tipo_pantalla, tipo_dia, columnas_entrenamiento)
                
                # 2. Predecir
                prediccion_raw = modelo.predict(df_real)[0]
                
                # 3. Mapear resultado a texto
                mapa_clases = {0: "MUY BAJA", 1: "BAJA", 2: "MODERADA", 3: "ALTA"}
                
                # Manejo robusto de la respuesta (texto o número)
                if isinstance(prediccion_raw, str):
                    prediccion = prediccion_raw.upper()
                    traduccion = {"LOW": "BAJA", "VERY LOW": "MUY BAJA", "MODERATE": "MODERADA", "HIGH": "ALTA"}
                    prediccion = traduccion.get(prediccion, prediccion)
                else:
                    prediccion = mapa_clases.get(prediccion_raw, "DESCONOCIDO")

                origen = "Modelo Random Forest (Real)"
                
            except Exception as e:
                # Si falla algo técnico, mostramos error pequeño pero usamos simulación para no detener la demo
                st.error(f"Error técnico: {e}")
                prediccion = "ERROR"
                origen = "Error en cálculo"
        else:
            # RESPALDO: Si no hay archivos .pkl
            origen = "Simulación (Archivos no encontrados)"
            if horas_pantalla < 2.0: prediccion = "ALTA"
            elif horas_pantalla < 3.5: prediccion = "MODERADA"
            elif horas_pantalla < 5.5: prediccion = "BAJA"
            else: prediccion = "MUY BAJA"

        # --- MOSTRAR RESULTADO VISUAL ---
        if prediccion != "ERROR":
            color_map = {"ALTA": "#2ecc71", "MODERADA": "#f1c40f", "BAJA": "#e67e22", "MUY BAJA": "#e74c3c"}
            color_final = color_map.get(prediccion, "#333")
            
            st.markdown(f"""
            <div style="background-color: {color_final}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;">
                <h1 style="margin:0; color: white;">{prediccion}</h1>
                <p style="margin:0; font-size: 12px; opacity: 0.8;">Fuente: {origen}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tarjeta de Recomendación (Estilo fijo para impresión/dark mode)
            st.markdown(f"""
            <style> @media print {{ .box {{ -webkit-print-color-adjust: exact !important; }} }} </style>
            <div class="box" style="padding: 15px; background-color: #f0f2f6; color: #333333; border-radius: 10px; border-left: 5px solid {color_final};">
                <h4 style="color: #333333; margin:0;">💡 Recomendación:</h4>
                <p style="color: #333333; margin-top: 5px;">
                    Basado en un perfil de <b>{edad} años</b> con uso <b>{tipo_pantalla}</b>: 
                    Se sugiere monitorear los tiempos de descanso y fomentar actividades fuera de línea.
                </p>
            </div>
            """, unsafe_allow_html=True)

with col_der:
    st.write("### 🔍 Datos Técnicos")
    # Información técnica discreta
    if modelo:
        st.caption("✅ Modelo: Random Forest (v1.0)")
        with st.expander("Ver Parámetros Internos"):
            st.code(modelo.get_params())
    else:
        st.caption("⚠️ Modelo no cargado (Modo Demo)")
