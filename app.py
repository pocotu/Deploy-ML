import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# Configuracion de la pagina
st.set_page_config(
    page_title="Predictor de Viajes - STI Cusco",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .card-dark {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    
    /* Resultado principal */
    .result-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .result-value {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .result-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Stats boxes */
    .stat-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-top: 0.25rem;
    }
    
    /* Info tag */
    .info-tag {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .info-tag-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .info-tag-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Boton personalizado */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }
    
    /* Inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #94a3b8;
        font-size: 0.8rem;
    }
    
    /* Rango de tiempo */
    .time-range {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .time-range-item {
        background: rgba(255,255,255,0.15);
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    
    /* Modelo info */
    .model-info {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #525252;
    }
    
    .model-info p {
        margin: 0.3rem 0;
    }
    
    /* Desplegable HTML nativo */
    .model-details {
        margin-top: 0.5rem;
    }
    
    .model-details summary {
        cursor: pointer;
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.5rem 0;
        list-style: none;
    }
    
    .model-details summary::-webkit-details-marker {
        display: none;
    }
    
    .model-details summary::before {
        content: '+ ';
        font-weight: 600;
    }
    
    .model-details[open] summary::before {
        content: '- ';
    }
    
    .model-details summary:hover {
        color: #3b82f6;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Rutas a los archivos del modelo (mismo directorio)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def cargar_modelo():
    """Carga el modelo y sus componentes"""
    try:
        modelo = joblib.load(os.path.join(MODEL_DIR, 'modelo_prediccion_transporte_cusco.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_transporte_cusco.pkl'))
        encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_vehicle.pkl'))
        vehicle_stats = joblib.load(os.path.join(MODEL_DIR, 'vehicle_stats.pkl'))
        return modelo, scaler, encoder, vehicle_stats, None
    except Exception as e:
        return None, None, None, None, str(e)

def obtener_categoria_tiempo(hora):
    """Determina la categoria de tiempo segun la hora"""
    if 6 <= hora < 8:
        return 'pico_manana'
    elif 8 <= hora < 12:
        return 'manana'
    elif 12 <= hora < 14:
        return 'almuerzo'
    elif 14 <= hora < 17:
        return 'tarde'
    elif 17 <= hora < 20:
        return 'pico_tarde'
    else:
        return 'noche'

def obtener_nombre_categoria(categoria):
    """Devuelve nombre legible de la categoria"""
    nombres = {
        'pico_manana': 'Hora Pico AM',
        'manana': 'Manana',
        'almuerzo': 'Medio Dia',
        'tarde': 'Tarde',
        'pico_tarde': 'Hora Pico PM',
        'noche': 'Noche'
    }
    return nombres.get(categoria, categoria)

def formato_tiempo(minutos):
    """Convierte minutos a formato legible (Xh Ym)"""
    horas = int(minutos // 60)
    mins = int(minutos % 60)
    if horas > 0:
        return f"{horas}h {mins}min"
    return f"{mins} min"

def preparar_features(lap, fecha_hora, vehicle_id, vehicle_data, encoder, global_mean=165, global_std=12, global_count=100):
    """Prepara las features para el modelo"""
    
    hora = fecha_hora.hour
    dia_semana = fecha_hora.weekday()
    dia = fecha_hora.day
    mes = fecha_hora.month
    es_fin_semana = 1 if dia_semana >= 5 else 0
    
    try:
        vehicle_encoded = encoder.transform([vehicle_id])[0]
    except:
        vehicle_encoded = 0
    
    if isinstance(vehicle_data, dict):
        if vehicle_id in vehicle_data:
            stats = vehicle_data[vehicle_id]
            vehicle_mean = stats['mean']
            vehicle_std = stats['std']
            vehicle_count = stats['count']
        else:
            vehicle_mean = global_mean
            vehicle_std = global_std
            vehicle_count = global_count
    else:
        if vehicle_id in vehicle_data.index:
            stats = vehicle_data.loc[vehicle_id]
            vehicle_mean = stats['mean']
            vehicle_std = stats['std']
            vehicle_count = stats['count']
        else:
            vehicle_mean = global_mean
            vehicle_std = global_std
            vehicle_count = global_count
    
    categoria = obtener_categoria_tiempo(hora)
    time_categories = {
        'time_afternoon': 1 if categoria == 'tarde' else 0,
        'time_evening_peak': 1 if categoria == 'pico_tarde' else 0,
        'time_lunch': 1 if categoria == 'almuerzo' else 0,
        'time_morning': 1 if categoria == 'manana' else 0,
        'time_morning_peak': 1 if categoria == 'pico_manana' else 0,
        'time_night': 1 if categoria == 'noche' else 0
    }
    
    features = pd.DataFrame([{
        'lap': lap,
        'hour': hora,
        'day_of_week': dia_semana,
        'day': dia,
        'month': mes,
        'is_weekend': es_fin_semana,
        'vehicle_encoded': vehicle_encoded,
        'vehicle_mean_duration': vehicle_mean,
        'vehicle_std_duration': vehicle_std,
        'vehicle_trip_count': vehicle_count,
        **time_categories
    }])
    
    return features

# ============== INTERFAZ ==============

# Header
st.markdown("""
<div class="main-header">
    <h1>Predictor de Duracion de Viajes</h1>
    <p>Sistema de Transporte Inteligente - Cusco</p>
</div>
""", unsafe_allow_html=True)

# Cargar modelo
modelo, scaler, encoder, vehicle_stats, error = cargar_modelo()

if error:
    st.markdown(f"""
    <div class="card">
        <span class="status-badge status-error">Error al cargar modelo</span>
        <p style="margin-top: 1rem; color: #525252;">{error}</p>
        <p style="color: #94a3b8; font-size: 0.85rem;">Verifica que los archivos .pkl esten en la carpeta del proyecto.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Extraer datos de vehicle_stats
if isinstance(vehicle_stats, dict) and 'vehicle_stats' in vehicle_stats:
    vehicle_data = vehicle_stats['vehicle_stats']
    global_mean = vehicle_stats.get('global_mean', 165)
    global_std = vehicle_stats.get('global_std', 12)
    global_count = vehicle_stats.get('global_count', 100)
else:
    vehicle_data = vehicle_stats
    global_mean, global_std, global_count = 165, 12, 100

if isinstance(vehicle_data, dict):
    vehiculos_disponibles = list(vehicle_data.keys())
else:
    vehiculos_disponibles = list(vehicle_data.index)

# Status del modelo
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <span class="status-badge status-success">Modelo activo</span>
</div>
""", unsafe_allow_html=True)

# Formulario de entrada
st.markdown('<p class="card-title">Configurar Viaje</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    vehiculo = st.selectbox(
        "Vehiculo",
        options=vehiculos_disponibles,
        help="ID del vehiculo"
    )
    
    fecha = st.date_input(
        "Fecha",
        value=datetime.now()
    )

with col2:
    lap = st.number_input(
        "Vuelta del dia",
        min_value=1,
        max_value=50,
        value=1,
        help="Numero de recorrido del vehiculo hoy (1 = primer viaje del dia)"
    )
    
    hora = st.time_input(
        "Hora de Salida",
        value=datetime.now().time(),
        help="Hora de inicio del viaje"
    )

fecha_hora = datetime.combine(fecha, hora)

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Info del vehiculo seleccionado
categoria_tiempo = obtener_categoria_tiempo(fecha_hora.hour)
dia_semana = fecha_hora.weekday()
dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']

# Stats del vehiculo
st.markdown('<p class="card-title">Datos del Vehiculo</p>', unsafe_allow_html=True)

if isinstance(vehicle_data, dict) and vehiculo in vehicle_data:
    stats = vehicle_data[vehiculo]
    promedio_fmt = formato_tiempo(stats['mean'])
    st.markdown(f"""
    <div class="card-dark">
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div>
                <p style="color: #64748b; font-size: 0.75rem; margin: 0; text-transform: uppercase;">Duracion promedio</p>
                <p style="font-size: 1.5rem; font-weight: 600; color: #1e293b; margin: 0;">{promedio_fmt}</p>
            </div>
            <div>
                <p style="color: #64748b; font-size: 0.75rem; margin: 0; text-transform: uppercase;">Viajes registrados</p>
                <p style="font-size: 1.5rem; font-weight: 600; color: #1e293b; margin: 0;">{int(stats['count'])}</p>
            </div>
            <div>
                <p style="color: #64748b; font-size: 0.75rem; margin: 0; text-transform: uppercase;">Variacion tipica</p>
                <p style="font-size: 1.5rem; font-weight: 600; color: #1e293b; margin: 0;">+/- {stats['std']:.0f} min</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card-dark">
        <p style="color: #64748b; margin: 0;">Sin datos historicos para este vehiculo</p>
    </div>
    """, unsafe_allow_html=True)

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Boton de prediccion
if st.button("Calcular Duracion Estimada", type="primary", use_container_width=True):
    
    features = preparar_features(lap, fecha_hora, vehiculo, vehicle_data, encoder, global_mean, global_std, global_count)
    features_scaled = scaler.transform(features)
    prediccion = modelo.predict(features_scaled)[0]
    
    # Formato legible
    tiempo_fmt = formato_tiempo(prediccion)
    min_fmt = formato_tiempo(max(0, prediccion - 4.4))
    max_fmt = formato_tiempo(prediccion + 4.4)
    
    # Resultado principal
    st.markdown(f"""
    <div class="result-card">
        <div class="result-value">{tiempo_fmt}</div>
        <div class="result-label">duracion estimada</div>
        <div class="time-range">
            <div class="time-range-item">Min: {min_fmt}</div>
            <div class="time-range-item">Max: {max_fmt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hora de llegada
    hora_llegada = fecha_hora + timedelta(minutes=prediccion)
    
    # Obtener promedio del vehiculo para comparar
    if isinstance(vehicle_data, dict) and vehiculo in vehicle_data:
        promedio_vehiculo = vehicle_data[vehiculo]['mean']
    else:
        promedio_vehiculo = global_mean
    
    # Interpretacion basada en comparacion con el promedio del vehiculo
    diferencia = prediccion - promedio_vehiculo
    if diferencia < -10:
        tag_class = "info-tag-success"
        mensaje = "Mas rapido de lo usual"
    elif diferencia <= 10:
        tag_class = "info-tag"
        mensaje = "Tiempo normal"
    else:
        tag_class = "info-tag-warning"
        mensaje = "Mas lento de lo usual"
    
    st.markdown(f"""
    <div class="card-dark" style="margin-top: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Llegada estimada</p>
                <p style="font-size: 1.25rem; font-weight: 600; color: #1e293b; margin: 0.25rem 0 0 0;">
                    {hora_llegada.strftime('%H:%M')}
                </p>
            </div>
            <span class="info-tag {tag_class}">{mensaje}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Info del modelo (desplegable HTML nativo)
st.markdown("""
<div class="divider"></div>
<details class="model-details">
    <summary>Ver informacion del modelo</summary>
    <div class="model-info" style="margin-top: 1rem;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <p><strong>Algoritmo:</strong> Random Forest</p>
            <p><strong>Validacion:</strong> TimeSeriesSplit</p>
            <p><strong>RMSE:</strong> 4.38 min</p>
            <p><strong>Features:</strong> 16 variables</p>
            <p><strong>R2:</strong> 0.858</p>
            <p><strong>Precision:</strong> 86%</p>
        </div>
    </div>
</details>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Proyecto de Aprendizaje Automatico</p>
    <p>UNSAAC 2025 - II</p>
</div>
""", unsafe_allow_html=True)
