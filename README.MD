# Estación 2: Análisis Visual de Calidad

Este repositorio contendra el código y los recursos para resolver el problema de clasificación de calidad visual de cajas con frutas a partir de imágenes utilizando **Python**. La función principal del sistema es estimar la calidad visual de un lote evaluando su madurez, el color y los defectos visibles.

**Equipo:** Sebastian Arredondo, Maximiliano Cantuarias y Cristian Vega

## Actividades Principales

* Recibir la imagen previamente validada por la estación 1.
* Definir categorías visuales simples para la clasificación: calidad alta, media, baja, defectos visibles o revisión.
* Construir y etiquetar un dataset de imágenes.
* Entrenar un modelo de visión por computador o un clasificador supervisado en Python.
* Evaluar el modelo con datos de prueba y analizar los resultados con una matriz de confusión.
* Entregar la calidad visual estimada junto con su respectivo nivel de confianza.

## Conceptos Clave y Contenidos

El desarrollo de este proyecto abarca las siguientes áreas:

* Visión por computador
* Clasificación de imágenes
* División de datos: Entrenamiento, validación y prueba
* Métricas de desempeño
* Matriz de confusión
* Análisis de sobreajuste y generalización

## Aprendizajes Esperados

Al finalizar este proyecto, el equipo deberá ser capaz de:

* Explicar cómo se entrenó y se evaluó el modelo de clasificación.
* Interpretar los errores de clasificación.
* Relacionar las características visuales detectadas con la calidad comercial del producto.

## Entregables Mínimos

* Dataset visual estructurado y etiquetado.
* Modelo de clasificación visual (código fuente y pesos del modelo).
* Matriz de confusión generada a partir de las pruebas.
* Módulo de salida que retorne la calidad visual y el nivel de confianza de la predicción.

## Tecnologías Sugeridas

* **Lenguaje:** Python
* **Librerías principales:** OpenCV, TensorFlow / PyTorch, Scikit-learn, NumPy, Matplotlib/Seaborn.

## Plan de accion

Para desarrollar esta inteligencia artificial en Python, se recolectarán fotografías de naranjas desde una estación de captura inicial, las cuales serán redimensionadas y normalizadas entre 0 y 1 mediante librerías como OpenCV o TensorFlow. Luego, se aplicarán técnicas de aumento de datos (data augmentation) para simular variaciones de iluminación y evitar el sobreajuste, implementando una red neuronal convolucional (CNN) con modelos preentrenados que aseguren alta precisión. Finalmente, tras optimizar y evaluar el modelo dividiendo los datos en fases de prueba y entrenamiento. El despliegue de la IA aun esta en planificacion.