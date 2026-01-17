# 🥔 App de Diagnóstico de Enfermedades en Papa

Aplicación web interactiva profesional desarrollada en **Python** y **Streamlit** para el diagnóstico asistido por inteligencia artificial de enfermedades en hojas de papa. Utiliza modelos de **Deep Learning** (CNN, ResNet50) para clasificar imágenes y generar reportes técnicos detallados.

---

## 🚀 Características Principales

*   **Diagnóstico en Tiempo Real**: Sube una imagen de una hoja de papa y obtén un diagnóstico instantáneo con porcentaje de confianza.
*   **Clasificación Multiclase**: Identifica entre:
    *   ✅ **Sano**
    *   ⚠️ **Tizón Temprano** (Early Blight)
    *   🔥 **Tizón Tardío** (Late Blight)
*   **Recomendaciones de Tratamiento**: Proporciona guías de tratamiento específicas y medidas preventivas para cada enfermedad detectada.
*   **Análisis Comparativo de Modelos**: Módulo dedicado para comparar el rendimiento de diferentes arquitecturas (CNN Propia vs ResNet50 Transfer Learning).
*   **Mapas de Calor**: Visualización de zonas afectadas mediante procesamiento de imagen (HSV).
*   **Reportes PDF**: Generación automática de reportes técnicos descargables con métricas, matrices de confusión y gráficos estadísticos.

---

## 🛠️ Tecnologías Usadas

Este proyecto hace uso de un stack tecnológico moderno para ciencia de datos y desarrollo web:

*   **Frontend**: [Streamlit](https://streamlit.io/) (Interfaz de usuario interactiva)
*   **Deep Learning (Backend)**:
    *   [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
    *   CNN (Red Neuronal Convolucional Personalizada)
    *   ResNet50 (Transfer Learning)
*   **Procesamiento de Imágenes**:
    *   [OpenCV](https://opencv.org/) (Segmentación y preprocesamiento)
    *   Pillow (Manejo de imágenes)
*   **Análisis de Datos y Visualización**:
    *   NumPy & Pandas
    *   Matplotlib & Seaborn
    *   Scikit-learn (Métricas de evaluación)
*   **Reportes**:
    *   `pdfkit` y `wkhtmltopdf` (Generación de PDF desde HTML)

---

## 💻 Guía de Instalación

Sigue estos pasos para ejecutar la aplicación en tu entorno local.

### Prerrequisitos
*   **Python 3.10** o superior.
*   **wkhtmltopdf** (Solo necesario si deseas generar reportes PDF).
    *   Windows: [Descargar instalador](https://wkhtmltopdf.org/downloads.html) e instalar. Asegúrate de que la ruta en `app.py` coincida (`D:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe` o ajustala).

### Pasos

1.  **Clonar el repositorio**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd App_Diagnostico_Enfermades_Papa
    ```

2.  **Crear un entorno virtual (Recomendado)**
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En Mac/Linux:
    source venv/bin/activate
    ```

3.  **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Cómo Usar la Aplicación

1.  **Iniciar el servidor de Streamlit**
    Ejecuta el siguiente comando en tu terminal dentro de la carpeta del proyecto:
    ```bash
    streamlit run app.py
    ```

2.  **Abrir en el navegador**
    La aplicación se abrirá automáticamente en tu navegador predeterminado (usualmente en `http://localhost:8501`).

3.  **Navegación**:
    *   **Diagnóstico**: Sube una foto en la barra lateral para analizarla.
    *   **Guía de Enfermedades**: Consulta información educativa sobre los síntomas y tratamientos.
    *   **Análisis Comparativo**: Revisa las métricas técnicas de los modelos.
    *   **Reportes Técnicos**: Genera y descarga el PDF.

---

## 📂 Estructura del Proyecto

```text
App_Diagnostico_Enfermades_Papa/
├── app.py                 # Archivo principal de la aplicación Streamlit
├── requirements.txt       # Lista de dependencias
├── models/                # Archivos .h5 de los modelos entrenados
│   ├── best_potato_model.h5
│   └── ...
├── data/                  # Datos de prueba y ejemplos
│   ├── examples/          # Imágenes de ejemplo para la guía
│   ├── X_test.npy         # Datos de test para métricas
│   └── ...
├── reports/               # Carpeta para guardar reportes temporales (opcional)
└── README.md              # Documentación del proyecto
```

---

## 🔍 Detalles del Dataset

Se utilizó una versión curada del **PlantVillage Dataset**:
*   **Total de Imágenes**: 3,000+ (Balanceadas).
*   **Preprocesamiento**: Redimensionamiento a 256x256, normalización y aumento de datos (Data Augmentation).
*   **Métricas del Mejor Modelo**:
    *   **Precisión (Accuracy)**: >98%
    *   **Sensibilidad (Recall)**: >97%
    *   **F1-Score**: >98%

---

## 📄 Notas Adicionales

*   **Configuración de PDF**: Si tienes problemas generando el PDF, verifica la variable `RUTA_WKHTMLTOPDF` en el archivo `app.py` (línea 18) y asegúrate de que apunte a donde instalaste `wkhtmltopdf.exe`.
