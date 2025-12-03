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

# --- CARGAR MODELO REAL (ACTUALIZADO PARA CARPETA 'model') ---
@st.cache_resource
def cargar_modelo():
    try:
        # AQUI ESTA EL CAMBIO: Ahora busca dentro de la carpeta 'model/'
        model = joblib.load('model/modelo_random_forest.pkl')
        cols = joblib.load('model/columnas_modelo.pkl')
        return model, cols, "ÉXITO"
    except Exception as e:
        return None, None, str(e)

modelo, columnas_entrenamiento, estado_carga = cargar_modelo()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Perfil del Niño")
    
    if estado_carga == "ÉXITO":
        st.success("✅ Modelo IA Cargado (Desde carpeta /model)")
    else:
        st.error(f"⚠️ Error: No se encuentra el modelo en la carpeta 'model/'.")
        st.caption(f"Detalle: {estado_carga}")
    
    with st.expander("Información Demográfica", expanded=True):
        edad = st.slider('Edad (años)', 1, 18, 10)
        genero = st.selectbox('Género', ['Masculino', 'Femenino', 'Otro'])
    
    with st.expander("Hábitos Digitales", expanded=True):
        horas_pantalla = st.slider('Horas diarias de pantalla', 0.0, 12.0, 3.5, step=0.5)
        tipo_pantalla = st.selectbox('Contenido Principal', ['Educativo', 'Recreacional', 'Mixto'])
        tipo_dia = st.radio('Contexto', ['Día de semana', 'Fin de semana'])

# --- PROCESAMIENTO DE DATOS ---
def procesar_entrada_real(edad, genero, horas, tipo_p, tipo_d, columnas_modelo):
    data = {col: 0 for col in columnas_modelo}
    
    data['Age'] = edad
    data['Average Screen Time'] = horas
    if 'Sample Size' in data: data['Sample Size'] = 120 
    
    # One-Hot Encoding Manual
    if genero == 'Masculino' and 'Gender_Male' in data: data['Gender_Male'] = 1
    elif genero == 'Femenino' and 'Gender_Female' in data: data['Gender_Female'] = 1
    
    if tipo_p == 'Educativo' and 'Screen Time Type_Educational' in data: data['Screen Time Type_Educational'] = 1
    elif tipo_p == 'Recreacional' and 'Screen Time Type_Recreational' in data: data['Screen Time Type_Recreational'] = 1
    
    if tipo_dia == 'Fin de semana' and 'Day Type_Weekend' in data: data['Day Type_Weekend'] = 1
    elif tipo_dia == 'Día de semana' and 'Day Type_Weekday' in data: data['Day Type_Weekday'] = 1
    
    return pd.DataFrame([data])

# --- INTERFAZ PRINCIPAL ---
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader("📊 Resultado del Análisis")
    
    if st.button('🚀 EJECUTAR PREDICCIÓN CON IA'):
        
        if modelo is not None and columnas_entrenamiento is not None:
            try:
                df_real = procesar_entrada_real(edad, genero, horas_pantalla, tipo_pantalla, tipo_dia, columnas_entrenamiento)
                prediccion_raw = modelo.predict(df_real)[0]
                
                # Mapeo de resultados
                mapa_clases = {0: "MUY BAJA", 1: "BAJA", 2: "MODERADA", 3: "ALTA"}
                if isinstance(prediccion_raw, str):
                    prediccion = prediccion_raw.upper()
                    traduccion = {"LOW": "BAJA", "VERY LOW": "MUY BAJA", "MODERATE": "MODERADA", "HIGH": "ALTA"}
                    prediccion = traduccion.get(prediccion, prediccion)
                else:
                    prediccion = mapa_clases.get(prediccion_raw, "DESCONOCIDO")

                origen = "Modelo Random Forest (Real)"
                
            except Exception as e:
                st.error(f"Error en predicción: {e}")
                prediccion = "ERROR"
        else:
            # Respaldo Simulado
            origen = "Simulación (Modelo no encontrado)"
            if horas_pantalla < 2.0: prediccion = "ALTA"
            elif horas_pantalla < 3.5: prediccion = "MODERADA"
            elif horas_pantalla < 5.5: prediccion = "BAJA"
            else: prediccion = "MUY BAJA"

        # Visualización
        color_map = {"ALTA": "#2ecc71", "MODERADA": "#f1c40f", "BAJA": "#e67e22", "MUY BAJA": "#e74c3c"}
        color_final = color_map.get(prediccion, "#333")
        
        st.markdown(f"""
        <div style="background-color: {color_final}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h1 style="margin:0; color: white;">{prediccion}</h1>
            <p style="margin:0;">Fuente: {origen}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <style> @media print {{ .box {{ -webkit-print-color-adjust: exact !important; }} }} </style>
        <div class="box" style="margin-top: 20px; padding: 15px; background-color: #f0f2f6; color: #333; border-radius: 10px; border-left: 5px solid {color_final};">
            <h4 style="color: #333; margin:0;">💡 Recomendación:</h4>
            <p style="color: #333;">Basado en el perfil de <b>{edad} años</b>: Se sugiere monitorear los tiempos de descanso.</p>
        </div>
        """, unsafe_allow_html=True)

with col_der:
    st.write("### 🔍 Datos Técnicos")
    st.info(f"Status: {estado_carga}")
    if modelo:
        st.write("Hiperparámetros:")
        st.code(modelo.get_params())
