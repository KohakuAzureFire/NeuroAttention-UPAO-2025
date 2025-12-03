import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random 

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
        model = joblib.load('model/modelo_random_forest.pkl')
        cols = joblib.load('model/columnas_modelo.pkl')
        return model, cols
    except Exception as e:
        return None, None

modelo, columnas_entrenamiento = cargar_modelo()

# --- SIDEBAR ---
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
    
    if 'Average Screen Time (hours)' in data:
        data['Average Screen Time (hours)'] = horas
    else:
        data['Average Screen Time'] = horas
        
    if 'Age' in data: data['Age'] = edad
    if 'Sample Size' in data: data['Sample Size'] = 120 
    
    # One-Hot Encoding
    if genero == 'Masculino' and 'Gender_Male' in data: data['Gender_Male'] = 1
    elif genero == 'Otro':
        for col in data:
            if 'Gender' in col and 'Other' in col: data[col] = 1

    if tipo_p == 'Recreacional' and 'Screen Time Type_Recreational' in data: 
        data['Screen Time Type_Recreational'] = 1
    elif tipo_p == 'Mixto':
        if 'Screen Time Type_Total' in data: data['Screen Time Type_Total'] = 1

    if tipo_dia == 'Fin de semana' and 'Day Type_Weekend' in data: 
        data['Day Type_Weekend'] = 1
    
    return pd.DataFrame([data])

# --- BANCO DE CONOCIMIENTO EXPERTO (80 MENSAJES) ---
def obtener_recomendacion_dinamica(prediccion, tipo_pantalla, edad):
    
    banco_mensajes = {
        "ALTA": [
            "¡Excelente gestión! El equilibrio actual favorece la plasticidad cerebral y la concentración.",
            "Nivel óptimo. Las rutinas actuales están protegiendo la capacidad de atención sostenida del menor.",
            "Muy buen balance digital. Se sugiere mantener este ritmo y priorizar el sueño reparador de 8 horas.",
            "Resultado positivo. El tiempo de pantalla actual no parece interferir con el desarrollo cognitivo.",
            "Gestión ejemplar. Continúe fomentando actividades como la lectura y el deporte.",
            "¡Felicidades! Se detecta un entorno digital saludable que potencia el aprendizaje.",
            "El perfil indica una alta capacidad de enfoque. Mantenga los dispositivos fuera de la habitación al dormir.",
            "Equilibrio ideal. La relación entre tiempo online y offline es la recomendada por pediatras.",
            "Atención preservada. El niño tiene espacio mental suficiente para la creatividad y el aburrimiento constructivo.",
            "Excelente. Se recomienda mantener las reglas actuales y supervisar la calidad del contenido.",
            "Estado cognitivo favorable. El bajo estrés digital contribuye a un mejor rendimiento escolar.",
            "Gestión proactiva. Este nivel de uso permite un desarrollo socioemocional adecuado.",
            "Muy bien. Aproveche este estado de atención para introducir juegos de mesa o lógica.",
            "La higiene digital es correcta. No se observan riesgos inmediatos de dispersión.",
            "Control parental efectivo. Siga promoviendo el uso consciente de la tecnología.",
            "Nivel saludable. Recuerde que el ejemplo de los padres es el mejor maestro.",
            "Capacidad atencional intacta. Fomente el aprendizaje de un instrumento musical o arte.",
            "Gran trabajo. El tiempo libre se está invirtiendo adecuadamente en el mundo físico.",
            "Sin alertas. El cerebro del niño está descansado y listo para aprender.",
            "Perfecto. Mantenga la política de 'pantallas apagadas' durante las comidas familiares."
        ],
        "MODERADA": [
            "Nivel aceptable, pero no te confíes. Aplica la regla 20-20-20 para descansar la vista.",
            "Atención promedio. Podría mejorar significativamente si se reducen 30 minutos de pantalla al día.",
            "Zona de precaución. Monitorea si el niño presenta irritabilidad leve al retirar el dispositivo.",
            "Balanceado, pero se sugiere intercalar con más actividad física cardiovascular.",
            f"El uso {tipo_pantalla} es moderado, pero vigile la postura física al usar el dispositivo.",
            "Atención fluctuante. Intente establecer horarios fijos y predecibles para el uso de pantallas.",
            "Nivel medio. Se recomienda no usar pantallas durante los traslados en auto para fomentar la observación.",
            "Riesgo leve de fatiga. Asegúrese de que la iluminación de la pantalla no sea excesiva.",
            "Podría mejorar. Intente reemplazar una sesión digital por una conversación familiar.",
            "Estable, pero vigile el contenido. El algoritmo sugiere aumentar las horas de sueño.",
            "Atención estándar. Introduzca 'pausas activas' (estiramientos) entre sesiones digitales.",
            "Aceptable. Sin embargo, evite el 'multitasking' (usar TV y celular a la vez).",
            "Monitoreo sugerido. Verifique si el niño parpadea lo suficiente frente a la pantalla.",
            "Nivel intermedio. Fomente actividades que requieran paciencia, como armar rompecabezas.",
            "Atención parcial. Se sugiere crear una 'zona libre de Wi-Fi' en el hogar.",
            "Cuidado con la rutina. El uso moderado puede volverse excesivo sin supervisión.",
            "Balance frágil. Asegúrese de que las tareas escolares se hagan antes del tiempo de pantalla.",
            "Atención recuperable. Un fin de semana de 'detox' suave podría subir el nivel a ALTA.",
            "Aceptable, pero supervise los cambios de humor después de jugar.",
            "Regule el brillo. La fatiga ocular puede confundirse con falta de atención."
        ],
        "BAJA": [
            "Signos de dispersión. Es necesario establecer horarios de 'desconexión' más estrictos.",
            "El nivel de atención se ve comprometido. Reemplace una hora de pantalla por lectura en papel.",
            "Riesgo de fatiga digital. Se recomienda evitar pantallas estrictamente 2 horas antes de dormir.",
            f"Alerta de atención. Aunque el uso sea {tipo_pantalla}, el exceso de tiempo fragmenta la concentración.",
            "Atención reducida. El cerebro está recibiendo demasiada estimulación rápida (dopamina).",
            "Precaución. Se detectan patrones que podrían afectar el rendimiento académico.",
            "Necesita intervención. Reduzca el tiempo de pantalla gradualmente un 10% cada semana.",
            "Sobrecarga cognitiva. El niño podría tener dificultades para seguir instrucciones largas.",
            "Cuidado. El tiempo en pantalla está desplazando horas vitales de sueño o juego físico.",
            "Se sugiere acción. Implemente un 'toque de queda digital' a las 7:00 PM.",
            "Dispersión mental. Fomente actividades manuales (pintar, construir) para reconectar.",
            "Riesgo latente. La luz azul podría estar alterando los ritmos circadianos.",
            "Atención baja. Evite que el niño tenga televisor o computadora en su dormitorio.",
            "Alerta amarilla. Supervise si el niño pierde interés rápidamente en actividades offline.",
            "Desgaste atencional. Es vital reintroducir el aburrimiento sin tecnología como terapia.",
            "Sobrestimulación. El cerebro necesita silencio digital para procesar lo aprendido.",
            "Nivel deficiente. Considere usar aplicaciones de control parental para limitar el tiempo.",
            "Falta de enfoque. Priorice las conversaciones cara a cara sin celulares presentes.",
            "Riesgo académico. La memoria de trabajo podría estar saturada por el exceso de información.",
            "Se recomienda reducir el tiempo a la mitad durante los días escolares."
        ],
        "MUY BAJA": [
            "⚠️ Nivel crítico. Se sugiere una 'desintoxicación digital' inmediata de 48 horas.",
            "⚠️ Déficit de atención marcado. Es urgente establecer zonas libres de tecnología en el hogar.",
            "⚠️ Riesgo alto. El tiempo de exposición es excesivo para la edad; priorizar juegos manuales.",
            "⚠️ Alerta roja. Se recomienda consultar con un especialista si persisten problemas escolares.",
            "⚠️ Peligro de adicción. El sistema dopaminérgico podría estar sobrecargado.",
            "⚠️ Acción inmediata requerida. Retire los dispositivos y fomente el deporte al aire libre.",
            "⚠️ Impacto severo. La capacidad de concentración profunda está seriamente afectada.",
            "⚠️ Urgente: Establezca un 'ayuno de dopamina'. Cero pantallas por un fin de semana.",
            "⚠️ Riesgo de aislamiento. El mundo virtual está consumiendo demasiados recursos cognitivos.",
            "⚠️ Nivel preocupante. Es probable que el niño presente ansiedad si se le retira el móvil.",
            "⚠️ Intervención familiar necesaria. Todos en casa deben reducir el uso para dar el ejemplo.",
            "⚠️ Salud en riesgo. El sedentarismo asociado y la falta de atención son alarmantes.",
            "⚠️ Bloqueo cognitivo. El exceso de estímulos impide la consolidación de la memoria.",
            "⚠️ Situación límite. Busque actividades de 'Atención Plena' (Mindfulness) para niños.",
            "⚠️ Alerta máxima. El desarrollo de habilidades sociales podría estar estancado.",
            "⚠️ Desconexión total sugerida. Vuelva a lo básico: naturaleza, libros y deporte.",
            "⚠️ Crisis de atención. El niño 'escanea' la información en lugar de leerla o escucharla.",
            "⚠️ Riesgo conductual. Posible correlación con irritabilidad y falta de control de impulsos.",
            "⚠️ Prioridad absoluta: Recuperar el sueño y la actividad física antes de volver a usar pantallas.",
            "⚠️ El modelo detecta un patrón de uso compulsivo. Se requiere supervisión estricta constante."
        ]
    }
    
    # Seleccionar una frase al azar
    frases_disponibles = banco_mensajes.get(prediccion, ["Sin recomendación específica."])
    return random.choice(frases_disponibles)

# --- INTERFAZ PRINCIPAL ---
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader("📊 Resultado del Análisis")
    
    if st.button('🚀 EJECUTAR PREDICCIÓN CON IA'):
        
        if modelo is not None and columnas_entrenamiento is not None:
            try:
                df_real = procesar_entrada_real(edad, genero, horas_pantalla, tipo_pantalla, tipo_dia, columnas_entrenamiento)
                prediccion_raw = modelo.predict(df_real)[0]
                
                mapa_clases = {0: "MUY BAJA", 1: "BAJA", 2: "MODERADA", 3: "ALTA"}
                if isinstance(prediccion_raw, str):
                    prediccion = prediccion_raw.upper()
                    traduccion = {"LOW": "BAJA", "VERY LOW": "MUY BAJA", "MODERATE": "MODERADA", "HIGH": "ALTA"}
                    prediccion = traduccion.get(prediccion, prediccion)
                else:
                    prediccion = mapa_clases.get(prediccion_raw, "DESCONOCIDO")
                
                origen = "Modelo Random Forest" 

            except Exception as e:
                st.error(f"Error técnico: {e}")
                prediccion = "ERROR"
        else:
            # Respaldo Simulado
            if horas_pantalla < 2.0: prediccion = "ALTA"
            elif horas_pantalla < 3.5: prediccion = "MODERADA"
            elif horas_pantalla < 5.5: prediccion = "BAJA"
            else: prediccion = "MUY BAJA"
            origen = "Simulación"

        # --- MOSTRAR RESULTADO ---
        if prediccion != "ERROR":
            color_map = {"ALTA": "#2ecc71", "MODERADA": "#f1c40f", "BAJA": "#e67e22", "MUY BAJA": "#e74c3c"}
            color_final = color_map.get(prediccion, "#333")
            
            # Obtener recomendación aleatoria y dinámica
            texto_recomendacion = obtener_recomendacion_dinamica(prediccion, tipo_pantalla, edad)

            st.markdown(f"""
            <div style="background-color: {color_final}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;">
                <h1 style="margin:0; color: white;">{prediccion}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <style> @media print {{ .box {{ -webkit-print-color-adjust: exact !important; }} }} </style>
            <div class="box" style="padding: 15px; background-color: #f0f2f6; color: #333333; border-radius: 10px; border-left: 5px solid {color_final};">
                <h4 style="color: #333333; margin:0;">💡 Recomendación Experta:</h4>
                <p style="color: #333333; margin-top: 5px; font-size: 16px;">
                    {texto_recomendacion}
                </p>
            </div>
            """, unsafe_allow_html=True)

with col_der:
    st.write("### 🔍 Datos Técnicos")
    if modelo:
        st.caption("✅ Modelo: Random Forest (v1.0)")
        st.code("n_estimators=100\ncriterion='gini'\nrandom_state=42")
    else:
        st.caption("⚠️ Modelo no cargado (Modo Demo)")
