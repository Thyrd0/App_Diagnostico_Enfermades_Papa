import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import pandas as pd
from scipy import stats
import pdfkit
import base64 
from statsmodels.stats.contingency_tables import mcnemar

RUTA_WKHTMLTOPDF = r'D:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe' 

# Configuración de pdfkit
if RUTA_WKHTMLTOPDF and os.path.exists(RUTA_WKHTMLTOPDF):
    configuracion = pdfkit.configuration(wkhtmltopdf=RUTA_WKHTMLTOPDF)
else:
    # Si la ruta especificada no existe o es None, intenta buscarlo en el PATH del sistema
    configuracion = None 
    st.warning(
        "`wkhtmltopdf` no se encontró en la ruta especificada ni en el PATH del sistema. "
        "La generación de PDF podría fallar. Asegúrate de instalarlo y configurar `RUTA_WKHTMLTOPDF` si es necesario."
    )

# Mapeo de enfermedades
ENFERMEDADES = {
    0: 'Sano',
    1: 'Tizón Temprano',
    2: 'Tizón Tardío'
}
NOMBRES_CLASES = list(ENFERMEDADES.values())

# Configuración de la página Streamlit
st.set_page_config(
    page_title="Diagnóstico de Enfermedades en Papa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("🥔 Detector de Enfermedades en Hojas de Papa")
st.markdown("""
Sistema de inteligencia artificial para identificar **enfermedades comunes** en cultivos de papa 
y recomendar **tratamientos específicos**.
""")

# Cargar modelos y datos de prueba
@st.cache_resource
def cargar_todos_modelos():
    """Carga todos los modelos entrenados."""
    try:
        modelo_mejor = tf.keras.models.load_model('models/best_potato_model.h5')
        modelo_cnn_simple = tf.keras.models.load_model('models/potato_potatocnn_model.h5')
        modelo_resnet_tl = tf.keras.models.load_model('models/potato_resnet50_model.h5')
        return modelo_mejor, modelo_cnn_simple, modelo_resnet_tl
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}. Asegúrate de que los archivos .h5 estén en la carpeta 'models/'.")
        st.stop()

@st.cache_data
def cargar_datos_prueba_predicciones():
    """Carga X_test, y_test y las predicciones de cada modelo."""
    try:
        datos_prueba_X = np.load('data/X_test.npy')
        datos_prueba_y = np.load('data/y_test.npy')
        pred_cnn_simple = np.load('data/y_pred_potatocnn.npy')
        pred_resnet_tl = np.load('data/y_pred_resnet50.npy')
        pred_mejor_modelo = np.load('data/y_pred_best_model.npy')
        return datos_prueba_X, datos_prueba_y, pred_cnn_simple, pred_resnet_tl, pred_mejor_modelo
    except Exception as e:
        st.error(f"Error al cargar los datos de prueba o predicciones: {e}. Asegúrate de que los archivos .npy estén en la carpeta 'data/'.")
        st.stop()

# Cargar todos los modelos y datos al inicio de la aplicación
modelo_mejor, modelo_cnn_simple, modelo_resnet_tl = cargar_todos_modelos()
datos_prueba_X, datos_prueba_y, pred_cnn_simple, pred_resnet_tl, pred_mejor_modelo = cargar_datos_prueba_predicciones()

# Información de enfermedades y tratamientos
INFO_ENFERMEDADES = {
    0: {
        'nombre': 'Sano',
        'tratamiento': 'No se requiere tratamiento. Continúe con prácticas agrícolas preventivas.',
        'sintomas': 'Hojas verdes sin manchas ni decoloraciones',
        'color': '#8FBC8F'
    },
    1: {
        'nombre': 'Tizón Temprano',
        'tratamiento': '1. Fungicidas protectores (clorotalonil)\n2. Rotación de cultivos (3-4 años)\n3. Eliminar residuos infectados',
        'sintomas': 'Manchas concéntricas oscuras con halos amarillos',
        'color': '#DAA520'
    },
    2: {
        'nombre': 'Tizón Tardío',
        'tratamiento': '1. Fungicidas sistémicos (fosetil-Al)\n2. Reducir humedad foliar\n3. Destruir plantas gravemente afectadas',
        'sintomas': 'Lesiones acuosas que se vuelven necróticas',
        'color': '#A52A2A'
    }
}

# Preprocesamiento de imágenes
def preprocesar_imagen(imagen):
    img = np.array(imagen)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    
    # Segmentación de áreas enfermas
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rango_inferior = np.array([10, 50, 50])
    rango_superior = np.array([30, 255, 255])
    mascara = cv2.inRange(hsv, rango_inferior, rango_superior)
    img = cv2.bitwise_and(img, img, mask=mascara)
    
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Funciones para análisis
def calcular_metricas(y_real, y_pred, nombre_modelo):
    """Calcula métricas clave para un modelo."""
    reporte = classification_report(y_real, y_pred, target_names=NOMBRES_CLASES, output_dict=True)
    
    precision = reporte['accuracy']
    matriz_confusion = confusion_matrix(y_real, y_pred)
    
    sensibilidad_prom = reporte['macro avg']['recall']
    f1_prom = reporte['macro avg']['f1-score']

    mcc = matthews_corrcoef(y_real, y_pred)

    lista_especificidad = []
    for i in range(len(NOMBRES_CLASES)):
        vn = np.sum(np.delete(np.delete(matriz_confusion, i, axis=0), i, axis=1))
        fp = np.sum(matriz_confusion[:, i]) - matriz_confusion[i, i]
        if (vn + fp) > 0:
            lista_especificidad.append(vn / (vn + fp))
        else:
            lista_especificidad.append(0.0)
    especificidad_prom = np.mean(lista_especificidad)

    return {
        'Modelo': nombre_modelo,
        'Precisión': precision,
        'Sensibilidad (Avg)': sensibilidad_prom,
        'Especificidad (Avg)': especificidad_prom,
        'F1-Score (Avg)': f1_prom,
        'Coeficiente de Matthews': mcc
    }

def realizar_prueba_mcnemar(y_real, y_pred_modelo1, y_pred_modelo2, nombre_modelo1, nombre_modelo2):
    """Realiza la prueba de McNemar entre dos modelos."""
    correctos_modelo1 = (y_pred_modelo1 == y_real)
    correctos_modelo2 = (y_pred_modelo2 == y_real)

    n_b = np.sum(correctos_modelo1 & ~correctos_modelo2)
    n_c = np.sum(~correctos_modelo1 & correctos_modelo2)
    
    if (n_b + n_c) == 0:
        return {
            'Comparación': f'{nombre_modelo1} vs {nombre_modelo2}',
            'Estadística Chi2': 'N/A',
            'P-valor': 'N/A',
            'Conclusión': 'No hay desacuerdos en las predicciones entre modelos para McNemar.'
        }
        
    tabla_contingencia = [[0, n_b], [n_c, 0]]
    
    try:
        # Usar exact=False para la versión de chi-cuadrado para muestras grandes
        resultado = mcnemar(tabla_contingencia, exact=False)
        estadistico = resultado.statistic
        valor_p = resultado.pvalue
        conclusion = ""
        if valor_p < 0.05:
            conclusion = f"Existe una diferencia estadísticamente significativa en el rendimiento de {nombre_modelo1} y {nombre_modelo2} (p < 0.05)."
        else:
            conclusion = f"No hay una diferencia estadísticamente significativa en el rendimiento de {nombre_modelo1} y {nombre_modelo2} (p >= 0.05)."
        
        return {
            'Comparación': f'{nombre_modelo1} vs {nombre_modelo2}',
            'Estadística Chi2': f'{estadistico:.4f}',
            'P-valor': f'{valor_p:.4f}',
            'Conclusión': conclusion
        }
    except ValueError as e:
        return {
            'Comparación': f'{nombre_modelo1} vs {nombre_modelo2}',
            'Estadística Chi2': 'Error',
            'P-valor': 'Error',
            'Conclusión': f'Error al calcular McNemar: {e}'
        }

def graficar_matriz_confusion(y_real, y_pred, nombre_modelo, nombres_clases):
    """Genera y retorna la figura matplotlib de la matriz de confusión."""
    matriz_conf = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Greens',
                xticklabels=nombres_clases, yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión: {nombre_modelo}')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    fig.tight_layout()
    return fig

def matriz_confusion_a_buffer(y_real, y_pred, nombre_modelo, nombres_clases):
    """Genera la matriz de confusión como imagen PNG en un buffer."""
    fig = graficar_matriz_confusion(y_real, y_pred, nombre_modelo, nombres_clases)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    return buffer

# Sidebar
st.sidebar.title("Opciones")
modo_app = st.sidebar.selectbox(
    "Modo de Operación",
    ["Diagnóstico", "Guía de Enfermedades", "Análisis Comparativo", "Reportes Técnicos"]
)

# Módulo de diagnóstico
if modo_app == "Diagnóstico":
    st.header("🔍 Diagnóstico por Imagen")
    archivo_subido = st.file_uploader(
        "Suba una foto de hoja de papa", 
        type=["jpg", "jpeg", "png"]
    )
    
    if archivo_subido is not None:
        imagen = Image.open(archivo_subido)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(imagen, caption="Imagen subida", use_column_width=True)
            
        with col2:
            if st.button("Analizar", type="primary"):
                with st.spinner("Procesando imagen..."):
                    imagen_procesada = preprocesar_imagen(imagen)
                    prediccion = modelo_mejor.predict(imagen_procesada)
                    clase_predicha = np.argmax(prediccion[0])
                    confianza = np.max(prediccion[0]) * 100
                    enfermedad = INFO_ENFERMEDADES[clase_predicha]

                    # Resultado del modelo
                    st.markdown(f"""
                    <div style='border-left: 5px solid {enfermedad['color']}; padding: 10px;'>
                        <h3 style='color:{enfermedad['color']}'>{enfermedad['nombre']}</h3>
                        <p><b>Confianza:</b> {confianza:.1f}%</p>
                        <p><b>Síntomas típicos:</b> {enfermedad['sintomas']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Tratamiento
                    st.subheader("📋 Tratamiento Recomendado")
                    st.markdown(f"```\n{enfermedad['tratamiento']}\n```")

                    # Probabilidades
                    st.subheader("📊 Distribución de Probabilidades")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    probabilidades = prediccion[0] * 100
                    colores = [INFO_ENFERMEDADES[i]['color'] for i in range(len(INFO_ENFERMEDADES))]
                    barras = ax.bar(
                        [INFO_ENFERMEDADES[i]['nombre'] for i in range(len(INFO_ENFERMEDADES))],
                        probabilidades, color=colores
                    )
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel("Probabilidad (%)")
                    plt.ylim(0, 100)
                    for barra in barras:
                        altura = barra.get_height()
                        ax.annotate(f'{altura:.1f}%',
                                    xy=(barra.get_x() + barra.get_width() / 2, altura),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    st.pyplot(fig)
                    
# Módulo educativo
elif modo_app == "Guía de Enfermedades":
    st.header("📚 Guía Visual de Enfermedades")
    
    pestañas = st.tabs([INFO_ENFERMEDADES[i]['nombre'] for i in range(len(INFO_ENFERMEDADES))])
    
    for i, pestaña in enumerate(pestañas):
        with pestaña:
            enfermedad = INFO_ENFERMEDADES[i]
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {enfermedad['nombre']}")
                st.image(f"data/examples/{i}.jpg", 
                         caption=f"Ejemplo de {enfermedad['nombre']}")
            
            with col2:
                st.markdown("**Síntomas característicos:**")
                st.write(enfermedad['sintomas'])
                
                st.markdown("**Tratamiento recomendado:**")
                st.markdown(f"```\n{enfermedad['tratamiento']}\n```")
                
                if i != 0:
                    st.markdown("**Medidas preventivas:**")
                    st.write("- Usar semillas certificadas")
                    st.write("- Evitar riego por aspersión en horas cálidas")
                    st.write("- Mantener adecuado espaciamiento entre plantas")

elif modo_app == "Análisis Comparativo":
    st.header("📊 Análisis Comparativo de Modelos y Reportes")
    st.write("Aquí puedes ver las métricas de rendimiento y la comparación estadística de los modelos entrenados.")
    
    st.subheader("Métricas de Rendimiento")
    metricas_mejor = calcular_metricas(datos_prueba_y, pred_mejor_modelo, 'Mejor Modelo (Keras Tuner)')
    metricas_cnn = calcular_metricas(datos_prueba_y, pred_cnn_simple, 'CNN Simple')
    metricas_resnet = calcular_metricas(datos_prueba_y, pred_resnet_tl, 'ResNet50 (Transfer Learning)')

    df_metricas = pd.DataFrame([metricas_mejor, metricas_cnn, metricas_resnet])
    st.dataframe(df_metricas.set_index('Modelo').style.format({
        'Precisión': "{:.2%}",
        'Sensibilidad (Avg)': "{:.2%}",
        'Especificidad (Avg)': "{:.2%}",
        'F1-Score (Avg)': "{:.2%}",
        'Coeficiente de Matthews': "{:.4f}"
    }))

    st.subheader("Pruebas de McNemar")
    st.write("La prueba de McNemar compara si la proporción de desacuerdos entre dos clasificadores es significativamente diferente.")

    mcnemar_mejor_vs_cnn = realizar_prueba_mcnemar(datos_prueba_y, pred_mejor_modelo, pred_cnn_simple, 'Mejor Modelo', 'CNN Simple')
    mcnemar_mejor_vs_resnet = realizar_prueba_mcnemar(datos_prueba_y, pred_mejor_modelo, pred_resnet_tl, 'Mejor Modelo', 'ResNet50')
    mcnemar_cnn_vs_resnet = realizar_prueba_mcnemar(datos_prueba_y, pred_cnn_simple, pred_resnet_tl, 'CNN Simple', 'ResNet50')

    df_mcnemar = pd.DataFrame([mcnemar_mejor_vs_cnn, mcnemar_mejor_vs_resnet, mcnemar_cnn_vs_resnet])
    st.dataframe(df_mcnemar.set_index('Comparación'))

    st.subheader("Matrices de Confusión")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.pyplot(graficar_matriz_confusion(datos_prueba_y, pred_mejor_modelo, 'Mejor Modelo', NOMBRES_CLASES))
    with col2:
        st.pyplot(graficar_matriz_confusion(datos_prueba_y, pred_cnn_simple, 'CNN Simple', NOMBRES_CLASES))
    with col3:
        st.pyplot(graficar_matriz_confusion(datos_prueba_y, pred_resnet_tl, 'ResNet50', NOMBRES_CLASES))

    st.subheader("Generar Reporte PDF")
    st.write("Haz clic para generar un reporte PDF con todos los resultados del análisis.")

    if st.button("Generar y Descargar Reporte PDF"):
        if configuracion is None and not os.path.exists('/usr/bin/wkhtmltopdf') and not os.path.exists('/usr/local/bin/wkhtmltopdf'):
             st.error("`wkhtmltopdf` no está instalado o no se encontró en tu sistema. No se puede generar el PDF.")
        else:
            with st.spinner("Generando reporte PDF..."):
                # Capturar las figuras de matplotlib y convertirlas a base64
                buffer_cm_mejor = matriz_confusion_a_buffer(datos_prueba_y, pred_mejor_modelo, 'Mejor Modelo', NOMBRES_CLASES)
                cm_mejor_b64 = base64.b64encode(buffer_cm_mejor.getvalue()).decode('utf-8')

                buffer_cm_cnn = matriz_confusion_a_buffer(datos_prueba_y, pred_cnn_simple, 'CNN Simple', NOMBRES_CLASES)
                cm_cnn_b64 = base64.b64encode(buffer_cm_cnn.getvalue()).decode('utf-8')

                buffer_cm_resnet = matriz_confusion_a_buffer(datos_prueba_y, pred_resnet_tl, 'ResNet50', NOMBRES_CLASES)
                cm_resnet_b64 = base64.b64encode(buffer_cm_resnet.getvalue()).decode('utf-8')
                
                # Construir el contenido HTML
                contenido_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Reporte de Diagnóstico de Enfermedades en Papa</title>
                    <style>
                        body {{ font-family: sans-serif; margin: 40px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        img {{ max-width: 100%; height: auto; display: block; margin: 20px 0; }}
                        .container {{ page-break-inside: avoid; }}
                    </style>
                </head>
                <body>
                    <h1>Reporte de Diagnóstico de Enfermedades en Papa</h1>
                    <p>Fecha del Reporte: {pd.Timestamp.now().strftime('%d-%m-%Y %H:%M:%S')}</p>

                    <div class="container">
                        <h2>Métricas de Rendimiento de Modelos</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Modelo</th>
                                    <th>Precisión</th>
                                    <th>Sensibilidad (Avg)</th>
                                    <th>Especificidad (Avg)</th>
                                    <th>F1-Score (Avg)</th>
                                    <th>Coeficiente de Matthews</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Mejor Modelo (Keras Tuner)</td>
                                    <td>{metricas_mejor['Precisión']:.2%}</td>
                                    <td>{metricas_mejor['Sensibilidad (Avg)']:.2%}</td>
                                    <td>{metricas_mejor['Especificidad (Avg)']:.2%}</td>
                                    <td>{metricas_mejor['F1-Score (Avg)']:.2%}</td>
                                    <td>{metricas_mejor['Coeficiente de Matthews']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>CNN Simple</td>
                                    <td>{metricas_cnn['Precisión']:.2%}</td>
                                    <td>{metricas_cnn['Sensibilidad (Avg)']:.2%}</td>
                                    <td>{metricas_cnn['Especificidad (Avg)']:.2%}</td>
                                    <td>{metricas_cnn['F1-Score (Avg)']:.2%}</td>
                                    <td>{metricas_cnn['Coeficiente de Matthews']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>ResNet50 (Transfer Learning)</td>
                                    <td>{metricas_resnet['Precisión']:.2%}</td>
                                    <td>{metricas_resnet['Sensibilidad (Avg)']:.2%}</td>
                                    <td>{metricas_resnet['Especificidad (Avg)']:.2%}</td>
                                    <td>{metricas_resnet['F1-Score (Avg)']:.2%}</td>
                                    <td>{metricas_resnet['Coeficiente de Matthews']:.4f}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div class="container">
                        <h2>Pruebas de McNemar</h2>
                        {df_mcnemar.to_html(escape=False)}
                    </div>

                    <div class="container">
                        <h2>Matrices de Confusión</h2>
                        <h3>Mejor Modelo (Keras Tuner)</h3>
                        <img src="data:image/png;base64,{cm_mejor_b64}">
                        <h3>CNN Simple</h3>
                        <img src="data:image/png;base64,{cm_cnn_b64}">
                        <h3>ResNet50 (Transfer Learning)</h3>
                        <img src="data:image/png;base64,{cm_resnet_b64}">
                    </div>
                </body>
                </html>
                """
                # Generar el PDF
                try:
                    bytes_pdf = pdfkit.from_string(contenido_html, False, configuration=configuracion)
                    st.download_button(
                        label="Descargar Reporte PDF",
                        data=bytes_pdf,
                        file_name="Reporte_Diagnostico_Papa.pdf",
                        mime="application/pdf"
                    )
                    st.success("Reporte PDF generado exitosamente.")
                except Exception as e:
                    st.error(f"Error al generar el PDF: {e}. Asegúrate de que `wkhtmltopdf` esté correctamente instalado y su ruta configurada.")
                    st.info("Para Windows, la ruta puede ser `r'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'`.")
                    st.info("Para Linux, prueba con `sudo apt-get install wkhtmltopdf`.")

# Módulo técnico
elif modo_app == "Reportes Técnicos":
    st.subheader("Mapa de Calor de Enfermedades")
    st.write("""
    Este mapa de calor muestra las zonas más comunes donde aparecen síntomas de enfermedades en las hojas de papa,
    basado en nuestro conjunto de imágenes de referencia.
    """)
    
    # Cargar imágenes de ejemplo y generar mapa de calor
    try:
        # Lista de imágenes de ejemplo por enfermedad
        imagenes_ejemplo = [
            "data/examples/0.jpg",  # Sano
            "data/examples/1.jpg",  # Tizón Temprano
            "data/examples/2.jpg"   # Tizón Tardío
        ]
        
        # Procesar imágenes para mapa de calor
        mapas_calor = []
        for ruta_imagen in imagenes_ejemplo:
            img = cv2.imread(ruta_imagen)
            if img is None:
                st.error(f"No se pudo cargar la imagen: {ruta_imagen}")
                continue
                
            # Convertir a HSV para mejor detección de áreas enfermas
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Rangos para detectar áreas enfermas (ajustar según necesidad)
            rango_inferior = np.array([10, 50, 50])
            rango_superior = np.array([30, 255, 255])
            mascara = cv2.inRange(hsv, rango_inferior, rango_superior)
            
            # Suavizar el mapa de calor
            mapa_calor = cv2.GaussianBlur(mascara, (51, 51), 0)
            mapa_calor = cv2.applyColorMap(mapa_calor, cv2.COLORMAP_JET)
            
            # Superponer el mapa de calor con la imagen original
            superpuesto = cv2.addWeighted(img, 0.7, mapa_calor, 0.3, 0)
            mapas_calor.append((img, superpuesto))
        
        # Mostrar resultados en columnas
        if mapas_calor:
            columnas = st.columns(len(mapas_calor))
            for idx, (original, mapa_calor) in enumerate(mapas_calor):
                with columnas[idx]:
                    st.image([original, mapa_calor], 
                             caption=["Original", f"Mapa de Calor - {ENFERMEDADES[idx]}"], 
                             width=200)
        
        # Generar mapa de calor agregado
        st.subheader("Mapa de Calor Agregado")
        st.write("""
        Mapa de calor combinado que muestra los patrones comunes de infección en todas las imágenes analizadas.
        """)
        
        # Crear un mapa de calor agregado (ejemplo simplificado)
        mapa_calor_agregado = np.zeros((256, 256, 3), dtype=np.float32)
        contador = 0
        
        for ruta_imagen in imagenes_ejemplo:
            img = cv2.imread(ruta_imagen)
            if img is None:
                continue
                
            img = cv2.resize(img, (256, 256))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mascara = cv2.inRange(hsv, rango_inferior, rango_superior).astype(np.float32)
            mapa_calor_agregado[:,:,0] += mascara  # Canal rojo
            contador += 1
        
        if contador > 0:
            mapa_calor_agregado /= contador
            mapa_calor_agregado = cv2.normalize(mapa_calor_agregado, None, 0, 255, cv2.NORM_MINMAX)
            mapa_calor_agregado = mapa_calor_agregado.astype(np.uint8)
            mapa_calor_agregado = cv2.applyColorMap(mapa_calor_agregado, cv2.COLORMAP_JET)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(cv2.cvtColor(mapa_calor_agregado, cv2.COLOR_BGR2RGB))
            ax.set_title("Mapa de Calor Agregado de Áreas Afectadas")
            ax.axis('off')
            st.pyplot(fig)
            
            # Interpretación del mapa de calor
            st.markdown("""
            **Interpretación del Mapa de Calor:**
            - Las áreas rojas indican zonas donde las enfermedades aparecen con mayor frecuencia
            - Las áreas azules/verdes muestran tejido sano o menos afectado
            - Los patrones pueden revelar cómo se propagan las enfermedades en las hojas
            """)
            
    except Exception as e:
        st.error(f"Error al generar mapas de calor: {e}")
    st.header("📈 Rendimiento del Modelo")
    
    st.subheader("Comparación de Arquitecturas")
    st.image("reports/model_comparison.png")
    
    st.subheader("Matriz de Confusión")
    st.image("reports/confusion_matrix.png")
    
    st.subheader("Curvas de Aprendizaje")
    st.image("reports/learning_curves.png")
    
    st.markdown("""
    ### Métricas Clave:
    | Métrica             | Valor   |
    |-----------------------|---------|
    | Precisión Global      | 98.2%   |
    | Sensibilidad Promedio | 97.8%   |
    | F1-Score Promedio     | 98.0%   |
    | Tiempo Inferencia     | 110ms   |
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset:** PlantVillage Potato (3,000+ imágenes)  
**Modelo:** CNN Optimizado  
**Precisión:** 98.2% (test)  
**Actualizado:** Julio 2025
""")