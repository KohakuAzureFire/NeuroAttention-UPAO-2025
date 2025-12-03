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

# --- CARGAR MODELO REAL ---
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
    
    # Asignar variables numéricas (Nombre exacto del entrenamiento)
    if 'Average Screen Time (hours)' in data:
        data['Average Screen Time (hours)'] = horas
    else:
        data['Average Screen Time'] = horas
        
    if 'Age' in data: data['Age'] = edad
    if 'Sample Size' in data: data['Sample Size'] = 120 
    
    # One-Hot Encoding Manual
    if genero == 'Masculino' and 'Gender_Male' in data: 
        data['Gender_Male'] = 1
    elif genero == 'Otro':
        for col in data:
            if 'Gender' in col and 'Other' in col:
                data[col] = 1

    if tipo_p == 'Recreacional' and 'Screen Time Type_Recreational' in data: 
        data['Screen Time Type_Recreational'] = 1
    elif tipo_p == 'Mixto':
        if 'Screen Time Type_Total' in data: data['Screen Time Type_Total'] = 1

    if tipo_dia == 'Fin de semana' and 'Day Type_Weekend' in data: 
        data['Day Type_Weekend'] = 1
    
    return pd.DataFrame([data])

# --- LÓGICA DE RECOMENDACIONES DINÁMICAS ---
def obtener_recomendacion(prediccion, tipo_pantalla):
    if prediccion == "ALTA":
        return "¡Excelente gestión! El equilibrio actual favorece la concentración. Se sugiere mantener las rutinas de sueño actuales."
    elif prediccion == "MODERADA":
        return "Nivel aceptable. Se recomienda aplicar la regla 20-20-20 (descansar la vista cada 20 minutos) para evitar fatiga cognitiva."
    elif prediccion == "BAJA":
        if tipo_pantalla == "Educativo":
            return "Atención dispersa. Aunque el contenido es educativo, el exceso de tiempo está afectando. Introducir pausas activas cada 45 minutos."
        else:
            return "Signos de dispersión. Se recomienda reducir estrictamente el tiempo de ocio digital y fomentar deportes o lectura en papel."
    else: # MUY BAJA
        return "⚠️ **Alerta:** Nivel crítico. Se sugiere una 'desintoxicación digital' inmediata y establecer zonas libres de pantallas en el hogar."

# --- INTERFAZ PRINCIPAL ---
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader("📊 Resultado del Análisis")
    
    if st.button('🚀 EJECUTAR PREDICCIÓN CON IA'):
        
        if modelo is not None and columnas_entrenamiento is not None:
            try:
                df_real = procesar_entrada_real(edad, genero, horas_pantalla, tipo_pantalla, tipo_dia, columnas_entrenamiento)
                prediccion_raw = modelo.predict(df_real)[0]
                
                # Mapeo
                mapa_clases = {0: "MUY BAJA", 1: "BAJA", 2: "MODERADA", 3: "ALTA"}
                if isinstance(prediccion_raw, str):
                    prediccion = prediccion_raw.upper()
                    traduccion = {"LOW": "BAJA", "VERY LOW": "MUY BAJA", "MODERATE": "MODERADA", "HIGH": "ALTA"}
                    prediccion = traduccion.get(prediccion, prediccion)
                else:
                    prediccion = mapa_clases.get(prediccion_raw, "DESCONOCIDO")

            except Exception as e:
                st.error(f"Error técnico: {e}")
                prediccion = "ERROR"
        else:
            # Respaldo Simulado
            if horas_pantalla < 2.0: prediccion = "ALTA"
            elif horas_pantalla < 3.5: prediccion = "MODERADA"
            elif horas_pantalla < 5.5: prediccion = "BAJA"
            else: prediccion = "MUY BAJA"

        # --- MOSTRAR RESULTADO ---
        if prediccion != "ERROR":
            color_map = {"ALTA": "#2ecc71", "MODERADA": "#f1c40f", "BAJA": "#e67e22", "MUY BAJA": "#e74c3c"}
            color_final = color_map.get(prediccion, "#333")
            
            # 1. Obtenemos el texto dinámico
            texto_recomendacion = obtener_recomendacion(prediccion, tipo_pantalla)

            # 2. Caja del Resultado (SIN LA LÍNEA DE FUENTE)
            st.markdown(f"""
            <div style="background-color: {color_final}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;">
                <h1 style="margin:0; color: white;">{prediccion}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Tarjeta de Recomendación Inteligente
            st.markdown(f"""
            <style> @media print {{ .box {{ -webkit-print-color-adjust: exact !important; }} }} </style>
            <div class="box" style="padding: 15px; background-color: #f0f2f6; color: #333333; border-radius: 10px; border-left: 5px solid {color_final};">
                <h4 style="color: #333333; margin:0;">💡 Recomendación Personalizada:</h4>
                <p style="color: #333333; margin-top: 5px; font-size: 16px;">
                    {texto_recomendacion}
                </p>
            </div>
            """, unsafe_allow_html=True)

with col_der:
    st.write("### 🔍 Datos Técnicos")
    if modelo:
        st.caption("✅ Modelo: Random Forest (v1.0)")
        with st.expander("Ver Parámetros Internos"):
            st.code(modelo.get_params())
    else:
        st.caption("⚠️ Modelo no cargado (Modo Demo)")
