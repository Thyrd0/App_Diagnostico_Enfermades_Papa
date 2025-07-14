🥔 App de Diagnóstico de Enfermedades en Papa

Aplicación web interactiva desarrollada en Python + Streamlit para diagnosticar enfermedades en hojas de papa mediante modelos de aprendizaje profundo, y generar reportes técnicos en PDF con análisis estadísticos.

🚀 Características
Diagnóstico automático de enfermedades usando una imagen de hoja.

Clasificación entre:

✅ Hoja sana

⚠️ Tizón Temprano

🔥 Tizón Tardío

Recomendaciones de tratamiento personalizado.

Comparación de modelos (CNN, ResNet50, Keras Tuner).

Análisis estadístico de desempeño (prueba de McNemar).

Módulo de mapas de calor para identificar zonas más afectadas.

Generación de reportes técnicos en PDF con gráficas y métricas.

🧠 Tecnologías Usadas
Frontend: Streamlit

Backend: TensorFlow / Keras

Data: Dataset de papa de PlantVillage

Métricas: F1-Score, Especificidad, Sensibilidad, MCC

PDF: pdfkit + wkhtmltopdf

📁 Estructura del Proyecto
bash
Copiar
Editar
📦 App_Diagnostico_Enfermades_Papa
├── app.py                  # Aplicación principal Streamlit
├── models/                 # Modelos .h5 entrenados
│   ├── best_potato_model.h5
│   ├── potatocnn_model.h5
│   └── resnet50_model.h5
├── data/                   # Datos de prueba y predicciones .npy
│   ├── X_test.npy
│   ├── y_test.npy
│   └── y_pred_*.npy
├── data/examples/          # Imágenes de ejemplo para visualización
├── reports/                # Reportes y gráficos (curvas, confusión)
├── docs/                   # Imágenes de documentación (README)
└── README.md               # Documentación del proyecto

