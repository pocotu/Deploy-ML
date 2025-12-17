---
title: Predictor de Transporte Cusco
emoji: 🚌
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# Predictor de Duracion de Viajes - STI Cusco

Sistema inteligente de prediccion de tiempos de viaje para el transporte publico en Cusco, Peru.

## Descripcion

Aplicacion web que utiliza Machine Learning para predecir la duracion de viajes de transporte publico basandose en:
- Historial del vehiculo
- Hora del dia y categoria temporal
- Dia de la semana
- Numero de vuelta del dia

## Caracteristicas

- Prediccion en tiempo real de duracion de viajes
- Analisis historico por vehiculo
- Rangos de confianza (min/max estimados)
- Interfaz intuitiva y responsive
- Categorizacion temporal (hora pico, almuerzo, noche, etc.)

## Modelo

- Algoritmo: Random Forest Regressor
- RMSE: 4.38 minutos
- R2 Score: 0.858
- Precision: 86%
- Features: 16 variables predictoras
- Validacion: TimeSeriesSplit

## Uso

1. Selecciona el vehiculo de la flota
2. Ingresa el numero de vuelta del dia
3. Elige fecha y hora de salida
4. Obten la prediccion instantanea

## Dataset

Basado en datos del Sistema de Transporte Inteligente (STI) de Cusco con aproximadamente 1 millon de registros de monitoreo.

## Proyecto Academico

Desarrollado como parte del curso de Aprendizaje Automatico - UNSAAC 2025-II

Objetivo: Comparar algoritmos de ML en problemas reales del transporte publico.

## Tecnologias

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Joblib

## Licencia

MIT License
