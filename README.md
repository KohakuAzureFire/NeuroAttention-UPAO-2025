# 🧠 NeuroAttention Predictor

**Proyecto de Aprendizaje Estadístico - UPAO 2025**

## 📋 Descripción
Este sistema utiliza técnicas de Machine Learning (Random Forest) para analizar y predecir el impacto del tiempo de pantalla en la capacidad de atención de niños. El proyecto busca proporcionar una herramienta de alerta temprana para padres y educadores, integrando datos demográficos y hábitos digitales para generar una clasificación del nivel de atención.

## 🚀 Características
- **Interfaz Interactiva:** Desarrollada con Streamlit para una experiencia de usuario amigable y moderna.
- **Modelo Predictivo:** Clasificación multiclase (Atención Alta, Moderada, Baja, Muy Baja) basada en un dataset híbrido de 120 registros.
- **Visualización de Impacto:** Sistema de semáforo con recomendaciones personalizadas según el tipo de consumo digital.
- **Alertas Visuales:** Tarjetas de recomendación que se adaptan al resultado del análisis.

## 🛠️ Manual de Instalación y Despliegue

Asegúrate de tener **Python 3.8+** instalado. A continuación, elige tu sistema operativo y ejecuta el bloque de comandos completo en tu terminal para configurar y lanzar la aplicación en un solo paso.

### 💻 Opción A: Para Windows

```bash
# 1. Clonar repositorio (si no lo has hecho)
git clone [https://github.com/KohakuAzureFire/NeuroAttention-UPAO-2025.git](https://github.com/KohakuAzureFire/NeuroAttention-UPAO-2025.git)
cd NeuroAttention-UPAO-2025

# 2. Configuración, Instalación y Ejecución (Todo en uno)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
### 🐧 Opción A: Para Linux

```bash
# 1. Clonar repositorio (si no lo has hecho)
git clone [https://github.com/KohakuAzureFire/NeuroAttention-UPAO-2025.git](https://github.com/KohakuAzureFire/NeuroAttention-UPAO-2025.git)
cd NeuroAttention-UPAO-2025

# 2. Configuración, Instalación y Ejecución (Todo en uno)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
