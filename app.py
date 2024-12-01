import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import os
import shap



# Cargar modelos entrenados previamente
modelos_prestaciones = {
    "Aguinaldo": "modelo_aguinaldo.pkl",
    "Vacaciones con sueldo": "modelo_vacaciones.pkl",
    "Servicio Médico": "modelo_servicio_medico.pkl",
    "Utilidades": "modelo_utilidades.pkl",
    "Incapacidad con sueldo": "modelo_incap_sueldo.pkl",
    "AFORE": "modelo_afore.pkl",
    "Crédito para vivienda": "modelo_credito_vivienda.pkl",
}

modelos_salario = {
    "cualquier_discapacidad": "modelo_salario_cualquier_discapacidad.pkl",
    "discapacidad_ver": "modelo_salario_discapacidad_ver.pkl",
    "discapacidad_oir": "modelo_salario_discapacidad_oir.pkl",
    "discapacidad_caminar": "modelo_salario_discapacidad_caminar.pkl",
    "discapacidad_banarse": "modelo_salario_discapacidad_banarse.pkl",
    "discapacidad_hablar": "modelo_salario_discapacidad_hablar.pkl",
    "discapacidad_recordar": "modelo_salario_discapacidad_recordar.pkl",
}

def cargar_modelo(nombre_modelo):
    return load(nombre_modelo)

# Configuración de la app
st.set_page_config(
    page_title="Condiciones Laborales",
    page_icon="📊",
    layout="wide",
)

# Sidebar para navegación
st.sidebar.title("Navegación")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    ("Explicación y guía", "Modelos Clasificadores de Prestaciones", "Predicción de Salarios por Discapacidad")
)

# Sección: Explicación y guía
if seccion == "Explicación y guía":
    st.title("Explicación, guía, autores y referencias")
    st.markdown(
        """
       De acuerdo con el Banco Mundial 15% de la población global, 1000 millones de personas, sufren algún tipo de discapacidad, y la prevalencia de la discapacidad es mayor en países de desarrollo. En México al año 2020 estimamos que viven 6.03 millones de personas que cuentan con algún tipo de discapacidad, lo cual representa al 4.8% de la población del país. De esta población 5.5 millones están en edad productiva (mayores a 12 años) y de esta población solo el 28.97% es económicamente activa, lo cual es una cifra inferior al porcentaje de 55.49% de población no discapacitada económicamente activa.
      
       Existen investigaciones enfocadas a temas similares, Alonso & Albarrán (2010) hicieron un análisis sobre el mercado laboral español de las personas con discapacidad y en situación de dependencia, mediante un modelo logit binario calcularon probabilidades diferenciadas de que las Personas con Discapacidad (PcD) tengan trabajo, utilizando diferentes variables demográficas. Pérez (2012) analiza la problemática de la discapacidad en México, señalando cómo las transformaciones del mercado laboral han generado vulnerabilidad y exclusión social, especialmente para PcD, debido a la falta de protecciones sociales y la precariedad laboral en un entorno dominado por el sector informal. Rodríguez y García (2021) estimaron diferencias salariales para las PcD y encontraron que hay discriminación salarial. Keating, Keramat,  Waller & Hashmi (2022) analizaron la satisfacción en diferentes condiciones de trabajo de las PcD, encontrando evidencia estadística que tener alguna discapacidad reduce la satisfacción dentro de las condiciones de trabajo. 
      
       La literatura sobre el tema expone la precariedad del panorama laboral para las PcD en México. A pesar de que se reconoce legalmente su derecho al trabajo, la mayoría enfrenta exclusión social debido a la falta de protecciones estatales y la prevalencia de empleos informales y de baja productividad, aunado al hecho de que las empresas suelen no aprovechar incentivos fiscales para contratarlos, y muchas organizaciones civiles carecen de los recursos suficientes para garantizar su inserción laboral efectiva. Esto deja a las personas discapacitadas en una situación de vulnerabilidad constante (Vite Pérez, 2012).
        Las personas con discapacidad integradas al mercado laboral pueden enfrentar **trato diferenciado** en comparación con los trabajadores sin discapacidad, debido a varios factores:
        
        

        ### Objetivo de esta Aplicación
        Esta aplicación incluye modelos predictivos para estimar:

        - **La probabilidad de obtener Prestaciones laborales específicas .**
        - **Salarios por hora** para personas con diferentes tipos de discapacidades, comparando entre personas con y sin discapacidad.

        ### Cómo Usar esta Aplicación
        1. **Selecciona la Sección**: 
            - *Modelos Clasificadores de Prestaciones*: Predice si deberías recibir una prestación laboral específica (como aguinaldo o vacaciones con sueldo) según tus características.
            - *Predicción de Salarios por Discapacidad*: Calcula el salario por hora para personas con un tipo específico de discapacidad y lo compara con el de personas sin discapacidad.

        2. **Modelos Clasificadores de Prestaciones**:
            - En esta sección, elige la prestación que deseas analizar.
            - Llena la información requerida, como tu edad, género, nivel educativo, y si tienes alguna discapacidad.
            - Haz clic en **"Predecir Prestación"**.
            - El modelo te mostrará si, de acuerdo con tus características, deberías tener la prestación laboral seleccionada y la probabilidad estimada de obtenerla.

        3. **Predicción de Salarios por Discapacidad**:
            - Selecciona el tipo de discapacidad que deseas analizar.
            - Llena la información requerida, como tu edad, género, nivel educativo, y otros datos relevantes.
            - Haz clic en **"Predecir Salarios"**.
            - La aplicación mostrará:
                - **Salario con discapacidad**: El salario estimado por hora para una persona con la discapacidad seleccionada.
                - **Salario sin discapacidad**: El salario estimado por hora para una persona sin discapacidad.

        ### Interpretación de Resultados
        - **Modelos Clasificadores de Prestaciones**:
            - El resultado indica si deberías o no tener la prestación laboral seleccionada. Además, te da la probabilidad estimada (entre 0 y 1) de obtenerla. Por ejemplo, una probabilidad del 0.85 significa que hay un 85% de posibilidad de que tengas esa prestación.
        - **Predicción de Salarios por Discapacidad**:
            - Compara los salarios estimados con y sin discapacidad. Por ejemplo:
                - Salario con discapacidad: $25.77 por hora.
                - Salario sin discapacidad: $28.75 por hora.
            - Esto significa que, según el modelo, existe una diferencia de $3.02 por hora entre los dos grupos.

        ### Consideraciones
        - Los modelos están basados en datos del Censo  2020 y pueden no reflejar completamente todas las situaciones individuales.
        - Los resultados deben interpretarse como herramientas de apoyo y no como determinantes absolutos.
        
        """
    )
     # ¿De dónde vienen las estimaciones?
    st.markdown(
        """
        ## ¿De dónde vienen las estimaciones?
        Las estimaciones en esta aplicación están basadas en modelos avanzados de **Machine Learning** y **Regresión Lineal**:

        - **Predicciones de prestaciones laborales**: 
          Estas predicciones se hicieron utilizando un modelo clasificador de **XGBoost** (Extreme Gradient Boosting). Este algoritmo utiliza una secuencia de árboles de decisión optimizados que trabajan en conjunto para calcular la importancia de cada variable y su efecto sobre la probabilidad de obtener o no la prestación. Por ejemplo, variables como la edad, género, nivel educativo, región de residencia, y si la persona tiene una discapacidad influyen directamente en el resultado.
          
          XGBoost es particularmente robusto para este tipo de problemas, ya que puede manejar datos complejos y detectar relaciones no lineales entre las variables. Además, su capacidad para optimizar iterativamente garantiza predicciones precisas.

        - **Predicciones de salario por hora**:
          En este caso, se utilizó un modelo de **regresión lineal**. Este método calcula el salario esperado en función de las variables ingresadas por el usuario, asumiendo una relación lineal entre estas variables y el salario. Por ejemplo, la escolaridad acumulada o la región de trabajo tienen un peso significativo en las estimaciones del modelo.

        Estos modelos fueron entrenados utilizando datos históricos, lo que asegura que las predicciones estén basadas en patrones reales observados previamente.
        """
    )

    # Autores
    st.markdown(
        """
        ## Autores
        Este proyecto fue desarrollado por:
        """
    )

    # Mostrar fotos de los autores
    col1, col2, col3 = st.columns(3)
    
    # Andrés Salas Garza
    with col1:
        st.image("Andrés_Salas.jpeg", caption="Andrés Salas Garza", width=200)
        st.markdown("Estudiante de Economía, séptimo semestre, hizo concentracion en Economia Pública y Desarrollo Sostenible, Economía Aplicada y Ciencia de Datos.")
        st.markdown("[LinkedIn](http://www.linkedin.com/in/andres-salas-garza)")
    
    # Emiliano Tress Ramírez
    with col2:
        st.image("Emiliano_Tress.jpeg", caption="Emiliano Tress Ramírez", width=200)
        st.markdown("Estudiante de Economía, séptimo semestre, hizo Concentración en Economía y Finanzas y Concentración en Economía Aplicada y Ciencia de Datos.")
        st.markdown("[LinkedIn](http://www.linkedin.com/in/emilianotressr)")
    # Sergio Fernando Gutiérrez Gutiérrez
    with col3:
        st.image("Sergio_Fernando_Gutierrez.jpeg", caption="Sergio Fernando Gutiérrez Gutiérrez", width=200)
        st.markdown("Estudiante de Economía, séptimo semestre, pasante en Consejo Nuevo León de febrero d a junio del 2024 e hizo concentración en Economía Aplicadad y Ciencia de Datos.")
        st.markdown("[LinkedIn](www.linkedin.com/in/sergio-fernando-gutiérrez-gutiérrez-b47a00329)")
   # Soporte
    st.markdown(
    """
    ## Soporte

    Aquí puedes encontrar un pequeño documento que explica cómo utilizar esta herramienta, el origen de los datos utilizados, y el funcionamiento de la metodología aplicada en los modelos predictivos de esta aplicación.
    
    Para más detalles, puedes descargar el documento en formato PDF haciendo clic en el siguiente botón:
    """
    )

    # Ruta al archivo PDF
    pdf_path = "Soporte.pdf"

# Botón para descargar el PDF
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        st.download_button(
            label="Descargar PDF",
            data=pdf_data,
            file_name="Soporte",
            mime="application/pdf",
    )
    except FileNotFoundError:
        st.error(f"El archivo no se encontró en la ruta especificada: {pdf_path}. Verifica la ubicación.")

        # Referencias
    st.markdown(
    """
    ## Referencias

    Las predicciones de los modelos hechos en esta página fueron desarrollados y entrenados gracias a los datos del **Cuestionario Ampliado del Censo de Población y Vivienda 2020**, los modelos para estimación de salario y predicción con algoritmos clasificadores y el contexto se basaron e inspiraron en trabajos previos cuyas fuentes se muestran a continuación:

    - Albarrán-Lozano, I., & Alonso-González, P. (2010). Participación en el mercado laboral español de las personas con discapacidad y en situación de dependencia. *Papeles de Poblacion, 16(64)*.
    - Castro, N. R., Moreira, G. C., & da Silva, R. P. (2019). Wage structure differential and disability in Brazil — Underperformance or discrimination? *EconomiA, 20(3)*. [https://doi.org/10.1016/j.econ.2019.11.003](https://doi.org/10.1016/j.econ.2019.11.003)
    - Chen, J., Mao, S., & Yuan, Q. (2022). Salary prediction using random forest with fundamental features. [https://doi.org/10.1117/12.2628520](https://doi.org/10.1117/12.2628520)
    - Eichinger, F., & Mayer, M. (2022). Predicting Salaries with Random-Forest Regression. [https://doi.org/10.1007/978-3-031-18483-3_1](https://doi.org/10.1007/978-3-031-18483-3_1)
    - INEGI. (2020). Cuestionario ampliado. *Censo de Población y Vivienda 2020*. Disponible en [https://www.inegi.org.mx/](https://www.inegi.org.mx/)
    - INEGI (2021). *Censo de Población y Vivienda 2020*. Disponible en [https://www.inegi.org.mx/programas/ccpv/2020/#documentacion](https://www.inegi.org.mx/programas/ccpv/2020/#documentacion)
    - Otero, J. V. (2012). Descomposición Oaxaca-Blinder en modelos lineales y no lineales. Disponible en [https://www.uam.es/uam/media/doc/1606862171313/blinder-oaxaca.pdf](https://www.uam.es/uam/media/doc/1606862171313/blinder-oaxaca.pdf)
    - Rodriguez Perez, R. E., & García Alvarado, F. de J. (2020). DESIGUALDAD SALARIAL ENTRE TRABAJADORES CON Y SIN DISCAPACIDAD EN MÉXICO, ¿DISCRIMINACIÓN O MENOR PRODUCTIVIDAD? *Ensayos Revista de Economía, 39(2)*. [https://doi.org/10.29105/ensayos39.2-4](https://doi.org/10.29105/ensayos39.2-4)
    - Vite Pérez M. (2012). La discapacidad en México desde la vulnerabilidad social. *Polis (Vol. 8, Issue 2)*.
    - Zhang, L., Guo, Z., Tao, Q., Xiong, Z., & Ye, J. (2023). XGBoost-based short-term prediction method for power system inertia and its interpretability. *Energy Reports, 9*. [https://doi.org/10.1016/j.egyr.2023.04.065](https://doi.org/10.1016/j.egyr.2023.04.065)
    """
)
# Sección: Modelos Clasificadores de Prestaciones
elif seccion == "Modelos Clasificadores de Prestaciones":
    st.title("Modelos Clasificadores de Prestaciones")
    modelo_seleccionado = st.sidebar.selectbox("Selecciona un Modelo", list(modelos_prestaciones.keys()))
    modelo_path = modelos_prestaciones[modelo_seleccionado]
    modelo = cargar_modelo(modelo_path)

    # Entrada de datos
    st.header(f"Predicción para {modelo_seleccionado}")
    edad = st.number_input("Edad", min_value=18, max_value=99, value=30)
    mujer = st.selectbox("Género", options=["Hombre", "Mujer"])
    escoacum = st.number_input("Escolaridad Acumulada (en años)", min_value=0, max_value=30, value=12)
    afrodes_new = st.selectbox("¿Es afrodescendiente?", options=["No", "Sí"])
    hlengua_new = st.selectbox("¿Habla una lengua indígena?", options=["No", "Sí"])
    cualquier_discapacidad = st.selectbox("¿Tiene alguna discapacidad?", options=["No", "Sí"])
    region = st.selectbox(
        "Región de Residencia",
        options=["Centro", "Noroeste", "Noreste", "Occidente/Bajío", "Sur"]
    )
    sector = st.selectbox("Sector del Trabajo", options=["Terciario", "Primario", "Secundario"])
    localidad = st.selectbox(
        "Tamaño de Localidad (Población)",
        options=[
            "Menor de 2,500 habitantes", "2,500 a 14,999 habitantes", 
            "15,000 a 49,999 habitantes", "50,000 a 99,999 habitantes", 
            "100,000 o más habitantes"
        ]
    )

    entradas = pd.DataFrame([{
        "edad": edad,
        "mujer": 1 if mujer == "Mujer" else 0,
        "escoacum": escoacum,
        "afrodes_new": 1 if afrodes_new == "Sí" else 0,
        "hlengua_new": 1 if hlengua_new == "Sí" else 0,
        "cualquier_discapacidad": 1 if cualquier_discapacidad == "Sí" else 0,
        "noroeste": 1 if region == "Noroeste" else 0,
        "noreste": 1 if region == "Noreste" else 0,
        "occidente_bajio": 1 if region == "Occidente/Bajío" else 0,
        "sur": 1 if region == "Sur" else 0,
        "act_prim": 1 if sector == "Primario" else 0,
        "act_sec": 1 if sector == "Secundario" else 0,
        "loc_rural": 1 if localidad == "Menor de 2,500 habitantes" else 0,
        "loc_semirural": 1 if localidad == "2,500 a 14,999 habitantes" else 0,
        "loc_semiurbano": 1 if localidad == "15,000 a 49,999 habitantes" else 0,
        "loc_urbano": 1 if localidad == "50,000 a 99,999 habitantes" else 0,
    }])

    if st.button("Predecir Prestación"):
        prediccion = modelo.predict(entradas)[0]
        probabilidad = modelo.predict_proba(entradas)[0, 1]
        st.write(
            f"Predicción: {'Sí tienes la prestación' if prediccion == 1 else 'No tienes la prestación'} para **{modelo_seleccionado}**."
        )
        st.write(f"Probabilidad: {probabilidad:.2f}")
        

# Mapeo de nombres formales
nombre_formal_map = {
    "cualquier_discapacidad": "Cualquier Discapacidad",
    "discapacidad_ver": "Discapacidad Visual",
    "discapacidad_oir": "Discapacidad Auditiva",
    "discapacidad_caminar": "Discapacidad Motriz",
    "discapacidad_banarse": "Discapacidad para Bañarse",
    "discapacidad_hablar": "Discapacidad del Habla",
    "discapacidad_recordar": "Discapacidad Cognitiva"
}

# Invertir el mapeo para obtener el nombre interno desde el formal
nombre_interno_map = {v: k for k, v in nombre_formal_map.items()}

# Actualización de la sección: Predicción de Salarios por Discapacidad
elif seccion == "Predicción de Salarios por Discapacidad":
    st.title("Predicción de Salarios por Discapacidad")
    
    # Usar los nombres formales en el menú desplegable
    discapacidad_formal_seleccionada = st.sidebar.selectbox(
        "Tipo de Discapacidad",
        list(nombre_formal_map.values())
    )
    
    # Obtener el nombre interno correspondiente
    discapacidad_seleccionada = nombre_interno_map[discapacidad_formal_seleccionada]
    
    modelo_path = modelos_salario[discapacidad_seleccionada]
    modelo = cargar_modelo(modelo_path)

    # Entrada de datos
    st.header(f"Predicción del Salario por Hora para {discapacidad_formal_seleccionada}")
    edad = st.number_input("Edad", min_value=18, max_value=99, value=30)
    mujer = st.selectbox("Género", options=["Hombre", "Mujer"])
    escoacum = st.number_input("Escolaridad Acumulada (en años)", min_value=0, max_value=30, value=12)
    afrodes_new = st.selectbox("¿Es afrodescendiente?", options=["No", "Sí"])
    hlengua_new = st.selectbox("¿Habla una lengua indígena?", options=["No", "Sí"])
    region = st.selectbox(
        "Región de Residencia",
        options=["Centro", "Noroeste", "Noreste", "Occidente/Bajío", "Sur"]
    )
    sector = st.selectbox("Sector del Trabajo", options=["Terciario", "Primario", "Secundario"])
    localidad = st.selectbox(
        "Tamaño de Localidad (Población)",
        options=[
            "Menor de 2,500 habitantes", "2,500 a 14,999 habitantes", 
            "15,000 a 49,999 habitantes", "50,000 a 99,999 habitantes", 
            "100,000 o más habitantes"
        ]
    )

    discapacidades = {
        "cualquier_discapacidad": 0,
        "discapacidad_ver": 0,
        "discapacidad_oir": 0,
        "discapacidad_caminar": 0,
        "discapacidad_banarse": 0,
        "discapacidad_hablar": 0,
        "discapacidad_recordar": 0
    }
    discapacidades_con = discapacidades.copy()
    discapacidades_con[discapacidad_seleccionada] = 1

    entradas_con = pd.DataFrame([{
        "edad": edad,
        "mujer": 1 if mujer == "Mujer" else 0,
        "escoacum": escoacum,
        "afrodes_new": 1 if afrodes_new == "Sí" else 0,
        "hlengua_new": 1 if hlengua_new == "Sí" else 0,
        "noroeste": 1 if region == "Noroeste" else 0,
        "noreste": 1 if region == "Noreste" else 0,
        "occidente_bajio": 1 if region == "Occidente/Bajío" else 0,
        "sur": 1 if region == "Sur" else 0,
        "act_prim": 1 if sector == "Primario" else 0,
        "act_sec": 1 if sector == "Secundario" else 0,
        "loc_rural": 1 if localidad == "Menor de 2,500 habitantes" else 0,
        "loc_semirural": 1 if localidad == "2,500 a 14,999 habitantes" else 0,
        "loc_semiurbano": 1 if localidad == "15,000 a 49,999 habitantes" else 0,
        "loc_urbano": 1 if localidad == "50,000 a 99,999 habitantes" else 0,
        **discapacidades_con,
    }])

    entradas_sin = pd.DataFrame([{
        "edad": edad,
        "mujer": 1 if mujer == "Mujer" else 0,
        "escoacum": escoacum,
        "afrodes_new": 1 if afrodes_new == "Sí" else 0,
        "hlengua_new": 1 if hlengua_new == "Sí" else 0,
        "noroeste": 1 if region == "Noroeste" else 0,
        "noreste": 1 if region == "Noreste" else 0,
        "occidente_bajio": 1 if region == "Occidente/Bajío" else 0,
        "sur": 1 if region == "Sur" else 0,
        "act_prim": 1 if sector == "Primario" else 0,
        "act_sec": 1 if sector == "Secundario" else 0,
        "loc_rural": 1 if localidad == "Menor de 2,500 habitantes" else 0,
        "loc_semirural": 1 if localidad == "2,500 a 14,999 habitantes" else 0,
        "loc_semiurbano": 1 if localidad == "15,000 a 49,999 habitantes" else 0,
        "loc_urbano": 1 if localidad == "50,000 a 99,999 habitantes" else 0,
        **discapacidades,
    }])

    if st.button("Predecir Salarios"):
        prediccion_con = modelo.predict(entradas_con)[0]
        salario_con = np.exp(prediccion_con)

        prediccion_sin = modelo.predict(entradas_sin)[0]
        salario_sin = np.exp(prediccion_sin)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Salario con discapacidad", f"${salario_con:.2f} por hora")
        with col2:
            st.metric("Salario sin discapacidad", f"${salario_sin:.2f} por hora")

        # Crear una gráfica de barras
        st.subheader("Comparación de Salarios")
        import matplotlib.pyplot as plt

        # Datos para la gráfica
        categorias = ["Con discapacidad", "Sin discapacidad"]
        salarios = [salario_con, salario_sin]

        # Crear la gráfica
        fig, ax = plt.subplots()
        ax.bar(categorias, salarios, color=['blue', 'green'])
        ax.set_ylabel("Salario por hora (MXN)")
        ax.set_title("Comparación de Salarios por Discapacidad")

        # Agregar valores encima de las barras
        for i, v in enumerate(salarios):
            ax.text(i, v + 0.5, f"${v:.2f}", ha='center', fontsize=10)

        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)
