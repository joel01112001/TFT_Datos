Estudio de tecnologías de visión por computador para la localización e identificación de participantes en el conjunto de datos de SoccerNet Challenges 2025


Descripción del Proyecto

Este proyecto es una solución integral para el análisis de videos de fútbol. Utiliza modelos avanzados de visión por computadora para detectar jugadores y el balón en tiempo real, identificar los dorsales y colores de camiseta de los jugadores, y asignar la posesión del balón. El resultado es un video anotado con información clave sobre los jugadores y la dinámica del juego.

Características Principales

Detección de Objetos: Identifica jugadores y el balón en cada fotograma del video.

Reconocimiento de Dorsales: Utiliza modelos de procesamiento de imágenes (BLIP o PaliGemma) para leer los números de los dorsales de los jugadores.

Detección de Color de Camiseta: Determina el color de la camiseta de cada jugador para la asignación de equipos.

Asignación de Equipos y Roles: Asigna jugadores a equipos (definidos por el usuario) y roles específicos (portero, árbitro).

Seguimiento de Jugadores: Mantiene un seguimiento coherente de los jugadores a lo largo del video.

Detección de Posesión del Balón: Identifica qué jugador tiene la posesión del balón.

Anotación de Video: Genera un nuevo video con cuadros delimitadores, dorsales, nombres de equipo/jugador y el poseedor del balón.

Registro Histórico: Guarda un historial detallado de las detecciones por jugador y fotograma en un archivo JSON.

Modelos Utilizados

El proyecto aprovecha el poder de modelos de IA de última generación:

YOLOv11: Para la detección eficiente de jugadores y balones en el video.

BLIP / PaliGemma: Para el reconocimiento de dorsales y la inferencia de colores de camiseta a partir de las imágenes de los jugadores. El usuario puede seleccionar qué modelo usar.

Configuración y Requisitos
Para ejecutar este proyecto, necesitarás tener instaladas las siguientes dependencias.

Requisitos Previos
Python 3.8 o superior

GPU con soporte CUDA (recomendado para un rendimiento óptimo)

Instalación de Dependencias
Puedes instalar todas las dependencias necesarias utilizando pip:



pip install torch ultralytics opencv-python-headless numpy Pillow transformers facet
Nota: Si encuentras problemas con opencv-python-headless, prueba con opencv-python.

Uso
1. Configuración de Colores y Modelo
Al iniciar el script, se te pedirá que ingreses los nombres de los equipos y los colores de sus camisetas (en español), así como los colores para el árbitro y el portero. También deberás seleccionar el modelo de lenguaje visual a utilizar (BLIP o PaliGemma) para la detección de dorsales y colores.

2. Ejecución del Análisis
El script main.py es el punto de entrada principal. Por defecto, procesará un video llamado "prueba.mp4" y generará un video de salida llamado "prueba.mp4".


3. Archivos de Salida
Una vez que el script finaliza, se generarán los siguientes archivos:

BLIP_Soccernet_SNGS-186.mp4 (o el nombre de salida que hayas configurado): El video anotado con las detecciones y la información de los jugadores.

dorsales_detectados/: Una carpeta que contendrá imágenes recortadas de los dorsales detectados para su inspección.

new_player_id_images/: Una carpeta que contendrá imágenes de los jugadores cuando se les asigna una nueva ID por primera vez.

historial_detecciones_YYYYMMDD_HHMMSS.json: Un archivo JSON que registra un historial detallado de las detecciones por jugador en cada fotograma, incluyendo el dorsal y el color confirmados, así como las posiciones y colores detectados a lo largo del tiempo.

Estructura del Proyecto
main.py: El script principal que orquesta el análisis de video.

config.py: Contiene la configuración global, como la carga de modelos, la definición de colores de equipo y los parámetros de los modelos.

video_analysis.py: Maneja la lógica de detección de jugadores, reconocimiento de dorsales y colores, y seguimiento. También guarda el historial de detecciones.

video_writer.py: Se encarga de tomar el video original y las detecciones para generar el video anotado final.

players.json: (Opcional) Un archivo JSON que puede contener una base de datos de jugadores por equipo y dorsal para mostrar nombres específicos en el video anotado.
