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

📄 Reporte PDF
La app permite exportar un reporte en PDF que incluye:

Tabla de métricas de rendimiento

Pruebas estadísticas (McNemar)

Gráficos de matrices de confusión

Visualizaciones de calor y diagnóstico

🔍 Dataset
Se utilizó una versión personalizada del dataset de papa del proyecto PlantVillage:

3 clases: sano, tizón temprano y tizón tardío

3000+ imágenes balanceadas

Imágenes redimensionadas y segmentadas en HSV para mejorar detección




