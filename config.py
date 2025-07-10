import torch
from ultralytics import YOLO
import cv2

def get_max_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max_frames

COLOR_SPANISH_TO_ENGLISH = {
    'rojo': 'red',
    'azul': 'blue',
    'verde': 'green',
    'amarillo': 'yellow',
    'blanco': 'white',
    'negro': 'black',
    'naranja': 'orange',
    'morado': 'purple',
    'rosa': 'pink',
    'gris': 'gray',
    'marron': 'brown',
    'arbitro': 'referee'
}

COLOR_MAP_RGB = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'pink': (203, 192, 255),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'brown': (42, 42, 165),
    'referee': (0, 0, 0)
}

def get_valid_color(prompt):
    while True:
        color_input = input(prompt).strip().lower()
        english_color = COLOR_SPANISH_TO_ENGLISH.get(color_input)
        
        if english_color and english_color in COLOR_MAP_RGB:
            return english_color
        print(f"Color inválido. Opciones válidas: {', '.join(COLOR_SPANISH_TO_ENGLISH.keys())}")

def get_team_colors():
    print("Introduce los nombres y colores de los equipos. Colores válidos (en español):")
    print(", ".join(COLOR_SPANISH_TO_ENGLISH.keys()))

    team1_name = input("Nombre del equipo 1: ").strip()
    team1_color = get_valid_color(f"Color para {team1_name}: ")

    team2_name = input("Nombre del equipo 2: ").strip()
    team2_color = get_valid_color(f"Color para {team2_name}: ")

    referee_color = get_valid_color("Color para el árbitro (referee): ")
    goalkeeper_color = get_valid_color("Color para el portero (goalkeeper): ")
    
    return {
        team1_name: team1_color,
        team2_name: team2_color,
        'referee': referee_color,
        'goalkeeper': goalkeeper_color
    }

def select_model():
    while True:
        model_choice = input("Seleccione el modelo a usar:\n1. BLIP\n2. PaliGemma\nIngrese el número de su elección: ").strip()
        if model_choice == '1':
            return 'BLIP'
        elif model_choice == '2':
            return 'PaliGemma'
        else:
            print("Opción inválida. Por favor, ingrese 1 o 2.")

yolo_model = YOLO('yolo11m.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model.to(device)
torch.backends.cudnn.benchmark = True

use_model = select_model()

if use_model == 'PaliGemma':
    from transformers import AutoProcessor, AutoModelForVision2Seq
    paligemma_processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
    paligemma_model = AutoModelForVision2Seq.from_pretrained("google/paligemma2-3b-mix-224")
    paligemma_model.to(device)
elif use_model == 'BLIP':
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

MIN_CONFIDENCE = 0
MIN_CONFIRMATIONS = 1
DIST_THRESHOLD = 50

TEAM_COLORS = get_team_colors()

dorsal_confirmation_threshold = 5
min_detections_required = 3