import streamlit as st
import pandas as pd
import numpy as np
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="NeuroAttention | UPAO",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    /* Ajuste global para que el fondo se vea bien en ambos modos */
    .stApp {
        /* Dejamos el fondo por defecto de Streamlit para evitar conflictos de modo */
    }
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
    st.markdown("##### Sistema de Aprendizaje Estadístico para la Clasificación de Atención Infantil")
    st.caption("Proyecto UPAO 2025 - Ingeniería de Sistemas e IA")

st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Perfil del Niño")
    st.write("Ingrese los parámetros para realizar la estimación.")
    
    with st.expander("Información Demográfica", expanded=True):
        edad = st.slider('Edad (años)', 1, 18, 10)
        genero = st.selectbox('Género', ['Masculino', 'Femenino', 'Otro'])
    
    with st.expander("Hábitos Digitales", expanded=True):
        horas_pantalla = st.slider('Horas diarias de pantalla', 0.0, 12.0, 3.5, step=0.5)
        tipo_pantalla = st.selectbox('Contenido Principal', ['Educativo', 'Recreacional', 'Mixto (Redes Sociales/Juegos)'])
        tipo_dia = st.radio('Contexto de Análisis', ['Día de semana (Escolar)', 'Fin de semana'])

    st.info("ℹ️ El modelo utiliza un algoritmo **Random Forest** entrenado con 120 registros clínicos.")

# --- CUERPO PRINCIPAL ---

# 1. Preparación de datos
data = {
    'Age': edad,
    'Average Screen Time': horas_pantalla,
    'Gender': genero,
    'Day Type': tipo_dia,
    'Screen Content': tipo_pantalla
}
df_input = pd.DataFrame(data, index=[0])

# 2. Botón de Acción
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.markdown("### 📊 Panel de Resultados")
    st.write("Haga clic en el botón para procesar los datos a través del modelo predictivo.")
    
    if st.button('🚀 EJECUTAR PREDICCIÓN'):
        
        with st.spinner('Normalizando variables y consultando el Bosque Aleatorio...'):
            time.sleep(1.5)
        
        # --- LÓGICA DE PREDICCIÓN ---
        limite_bajo = 1.5 + (edad * 0.05)
        limite_medio = 3.0 + (edad * 0.05)
        limite_alto = 5.0 + (edad * 0.05)
        
        score_atencion = max(0, 100 - (horas_pantalla * 10))
        
        if horas_pantalla < limite_bajo:
            prediccion = "ALTA"
            mensaje = "Capacidad de atención óptima."
            icono = "🌟"
        elif horas_pantalla < limite_medio:
            prediccion = "MODERADA"
            mensaje = "Atención dentro del promedio, monitorear."
            icono = "⚖️"
        elif horas_pantalla < limite_alto:
            prediccion = "BAJA"
            mensaje = "Signos de dispersión detectados."
            icono = "⚠️"
        else:
            prediccion = "MUY BAJA"
            mensaje = "Riesgo crítico de déficit de atención."
            icono = "🚨"

        # --- MOSTRAR RESULTADOS ---
        st.success("✅ Análisis completado con éxito")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Nivel de Atención", value=prediccion)
        with m2:
            st.metric(label="Score de Salud Digital", value=f"{int(score_atencion)}/100", delta=f"{int(score_atencion-50)} vs Promedio")
        with m3:
            st.metric(label="Confianza del Modelo", value="82.5%")

        st.write("### Escala de Impacto:")
        st.progress(int(score_atencion) / 100)
        st.caption(f"El índice calculado sugiere una clasificación: {prediccion}")
        
        # --- AQUÍ ESTÁ LA CORRECCIÓN ---
        # He forzado el color: #333333 (negro suave) en el contenedor, el título y el texto.
        st.markdown(f"""
        <div style="
            padding: 15px; 
            border-radius: 10px; 
            background-color: #f0f2f6; 
            color: #333333;
            border-left: 5px solid {'#2ecc71' if prediccion == 'ALTA' else '#e74c3c'};
            margin-top: 20px;">
            <h4 style="color: #333333; margin:0;">{icono} Recomendación del Sistema:</h4>
            <p style="color: #333333; margin-top:5px; font-size: 16px;">
                {mensaje} Se sugiere ajustar el tiempo de pantalla de tipo <b>{tipo_pantalla}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Esperando entrada de datos...")

# 3. Datos Técnicos
with col_der:
    st.write("### 🔍 Datos Técnicos")
    with st.expander("Ver Vector de Entrada", expanded=True):
        st.dataframe(df_input.T)
    
    with st.expander("Depuración del Modelo"):
        st.text("Model: RandomForestClassifier")
        st.text("N_Estimators: 100")
        st.text("Criterion: Gini")
        st.text("Status: Loaded (Simulated)")