# Predicción de Churn en Telecomunicaciones

Este proyecto analiza la predicción de abandono de clientes (churn) en el sector de telecomunicaciones utilizando el dataset público "Telco Customer Churn". El objetivo es comparar el desempeño de tres modelos de clasificación supervisada: Random Forest, Regresión Logística y Árboles de Decisión.

## Dataset

- **Fuente:** Telco Customer Churn (público)
- **Características:** Información demográfica, uso de servicios y detalles contractuales de 7,043 clientes.
- **Variable objetivo:** Estado de churn (abandono).

## Metodología

Se entrenaron y evaluaron los siguientes modelos:
- **Regresión Logística**
- **Random Forest**
- **Árbol de Decisión**

Las métricas empleadas incluyen precisión (accuracy), F1-Score y área bajo la curva ROC (AUC). Se analizaron también la importancia de variables y coeficientes para la interpretabilidad.

## Resultados

- **Regresión Logística** obtuvo el mejor desempeño (AUC: 0.856, Precisión: 81.9%).
- **Random Forest** y **Árbol de Decisión** presentaron resultados competitivos, pero inferiores.
- Las variables más influyentes fueron el tipo de contrato, la antigüedad del cliente y los cargos mensuales.

## Conclusiones

El aprendizaje automático permite identificar clientes con alto riesgo de abandono, facilitando estrategias de retención más efectivas en empresas de telecomunicaciones.
