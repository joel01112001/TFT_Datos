import cv2
import torch
import numpy as np
from PIL import Image
import re
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional, List
import os
from config import *
from retinaface import RetinaFace
import json
from datetime import datetime

os.makedirs("dorsales_detectados", exist_ok=True)

def extract_color_from_response(response: str, color_keywords: list) -> Optional[str]:
    for color in sorted(color_keywords, key=lambda x: -len(x)):
        if re.search(rf'\b{color}\b', response):
            return color
    return None

def get_dorsal_with_BLIP(image_pil):
    try:
        inputs = processor(
            images=image_pil,
            text="number",
            return_tensors="pt",
            truncation=True,
            max_length=32
        ).to(device)

        generated_ids = blip_model.generate(
            **inputs,
            max_new_tokens=2,
            num_beams=5,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True
        )

        response = processor.batch_decode(
            generated_ids.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()
        number_match = re.search(r'\d+', response)
        if number_match:
            return number_match.group()
        else:
            return None
    except Exception as e:
        print(f"Error procesando dorsal: {e}")
        return None

def detect_player_color_with_BLIP(image_pil, id: int) -> Optional[str]:
    try:
        inputs = processor(
            images=image_pil,
            text="shirt color",
            return_tensors="pt",
            truncation=True,
            max_length=32
        ).to(device)

        generated_ids = blip_model.generate(
            **inputs,
            max_new_tokens=10,
            num_beams=5,
            early_stopping=True,
            return_dict_in_generate=True
        )

        response = processor.batch_decode(
            generated_ids.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip().lower()

        color_keywords = list(set(TEAM_COLORS.values()))
        print(color_keywords)
        detected = extract_color_from_response(response, color_keywords)
        print(response)
        return detected

    except Exception as e:
        print(f"Error detectando color para ID {id}: {e}")
        return None

def assign_team(color: str) -> Tuple[str, str]:
    if not color:
        return ('unknown', 'unknown')
    if color == TEAM_COLORS['referee']:
        return ('referee', color)
    if color == TEAM_COLORS['goalkeeper']:
        return ('goalkeeper', color)
    for team, team_color in TEAM_COLORS.items():
        if team != 'referee' and color == team_color:
            return (team, color)
    return ('unknown', color)

def get_dorsal_with_PaliGemma(image_pil, confidence_threshold: float = 0.0) -> Optional[str]:
    try:
        prompt = "<image>""¿Esta el jugador de espaldas?si esta de espaldas dime el dorsal"
        inputs = paligemma_processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
        outputs = paligemma_model.generate(
            **inputs, 
            max_new_tokens=5, 
            output_scores=True, 
            return_dict_in_generate=True
        )
        
        response = paligemma_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        score_tensor = outputs.scores 
        confidences = [torch.max(torch.nn.functional.softmax(s, dim=-1)).item() for s in score_tensor]
        avg_confidence = sum(confidences) / len(confidences)
        
        number_match = re.search(r'\d+', response)
        
        if number_match and avg_confidence >= confidence_threshold:
            return number_match.group()
        else:
            return None
    except Exception as e:
        print(f"Error procesando dorsal con PaliGemma: {e}")
        return None

def detect_player_color_with_PaliGemma(image_pil) -> Optional[str]:
    try:
        prompt = "<image>What is the color of the player's shirt?"
        inputs = paligemma_processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
        outputs = paligemma_model.generate(
            **inputs, 
            max_new_tokens=10, 
            output_scores=True, 
            return_dict_in_generate=True
        )

        response = paligemma_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].lower()
        score_tensor = outputs.scores
        confidences = [torch.max(torch.nn.functional.softmax(s, dim=-1)).item() for s in score_tensor]
        avg_confidence = sum(confidences) / len(confidences)

        if response:
            color_keywords = list(set(TEAM_COLORS.values()))
            detected = extract_color_from_response(response, color_keywords)
            print(f"{response}, {detected}, {color_keywords}")
            return detected
        else:
            print(f"❌ Color no confiable para ID {id}, confianza {avg_confidence:.2f}")
            return None
    except Exception as e:
        print(f"Error detectando color con PaliGemma para ID {id}: {e}")
        return None
def is_key_color_frame(frame_count, max_frames):
    return frame_count == 0 or frame_count == max_frames // 2 or frame_count == max_frames // 4 or frame_count == 3*max_frames // 4 or frame_count == max_frames - 1

def analyze_video(input_path, use_model):
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    player_tracker = {}
    dorsal_history = defaultdict(list)  
    color_history = defaultdict(list)  
    detection_cooldown = defaultdict(int)
    color_cooldown = defaultdict(int)
    color_detection_attempts = defaultdict(int)
    next_id = 1
    frame_data = defaultdict(list)  
    confirmed_colors = {}

    
    dorsal_confirmation = defaultdict(list)  
    confirmed_dorsals = {}  
    MAX_FRAMES = get_max_frames(input_path)
    fully_confirmed_ids = set()
    NEW_IDS_SAVE_DIR = "new_player_id_images"
    os.makedirs(NEW_IDS_SAVE_DIR, exist_ok=True)

    while frame_count < MAX_FRAMES:
        print(f"Frame {frame_count}")
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo_model(frame, imgsz=640, half=True, device='0' if torch.cuda.is_available() else 'cpu')[0]
        current_ids = []

        for box in results.boxes:
            if yolo_model.names[int(box.cls[0])] != 'person':
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            best_id, best_dist = None, float('inf')

            for pid, data in player_tracker.items():
                last_pos = data['pos']
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(last_pos))
                if dist < DIST_THRESHOLD and dist < best_dist:
                    best_dist = dist
                    best_id = pid
            if best_id is None:
                best_id = next_id
                next_id += 1
                
                
                roi_new_id = frame[y1:y2, x1:x2]
                if roi_new_id.size > 0: 
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_id_filename = os.path.join(NEW_IDS_SAVE_DIR, f"new_id_{best_id}_frame{frame_count}_{timestamp}.jpg")
                    cv2.imwrite(new_id_filename, roi_new_id)
                    print(f"📸 Nueva ID {best_id} detectada y guardada en {new_id_filename}")

            if best_id is None:
                best_id = next_id
                next_id += 1
            if best_id not in detection_cooldown:
                detection_cooldown[best_id] = 0
            if best_id not in confirmed_dorsals:
                confirmed_dorsals[best_id] = False
            if best_id not in dorsal_history:
                dorsal_history[best_id] = []
            if best_id not in color_history:
                color_history[best_id] = []
            player_tracker[best_id] = {'pos': (cx, cy), 'bbox': (x1, y1, x2, y2)}
            current_ids.append(best_id)
            
            if best_id in confirmed_dorsals and best_id in fully_confirmed_ids:
                continue


            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).resize((960,960))
            
            
            detected_dorsal = None
            
            
            if detection_cooldown[best_id] <= 0 :
                if use_model == 'BLIP':
                    detected_dorsal = get_dorsal_with_BLIP(pil_img)
                elif use_model == 'PaliGemma':
                    detected_dorsal = get_dorsal_with_PaliGemma(pil_img, confidence_threshold=0.0)
                
                if detected_dorsal is not None :
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"dorsales_detectados/frame{frame_count}_id{best_id}_dorsal{detected_dorsal}_{timestamp}.jpg"
                    cv2.imwrite(filename, roi)
                    print(f"✅ Imagen guardada: {filename}")
                    
                    if best_id in confirmed_dorsals:
                        if detected_dorsal == confirmed_dorsals[best_id]:
                            
                            pass
                        else:
                            
                            
                            dorsal_history[best_id].append(detected_dorsal)
                            
                    else:
                        
                        dorsal_history[best_id].append(detected_dorsal)
                    detection_cooldown[best_id] = 15
                else:
                    
                    detection_cooldown[best_id] = 3
                    
            if best_id not in confirmed_colors:
                if use_model == 'BLIP':
                    detected_color = detect_player_color_with_BLIP(pil_img, best_id)
                elif use_model == 'PaliGemma':
                    detected_color = detect_player_color_with_PaliGemma(pil_img)

                if detected_color:
                    color_history[best_id].append(detected_color)
                    color_cooldown[best_id] = 1
                    confirmed_colors[best_id] = detected_color 
                else:   
                    color_cooldown[best_id] = 15 

                    

                        
                    
                
                
            if detected_dorsal:
                dorsal_counts = Counter(dorsal_history[best_id])
                
                most_common = dorsal_counts.most_common(1)
                if most_common:
                    most_common_dorsal, freq = most_common[0]
                else:
                    most_common_dorsal, freq = None, 0
                print(f"Historial ID {best_id}: {most_common_dorsal,freq}")
                if freq >= 3:
                    confirmed_dorsals[best_id] = most_common_dorsal
                    
                    fully_confirmed_ids.add(best_id)
                    print(f"✅ ID {best_id} completamente confirmado: dorsal {confirmed_dorsals[best_id]}")
            
                       
            if detection_cooldown[best_id] > 0:
                detection_cooldown[best_id] -= 1
            
            
            frame_entry = {
                'frame': frame_count,
                'dorsal': detected_dorsal if detected_dorsal is not None else -1,
                'posicion': (x1, y1, x2, y2),
                'color': color_history[best_id] if color_history[best_id] is not None else 'unknonw'
            }
            frame_data[best_id].append(frame_entry)
        
        frame_count += 1 
    
    cap.release()
    
    
    player_colors = {}
    for pid, colors in color_history.items():
        if pid in confirmed_colors:
            player_colors[pid] = assign_team(confirmed_colors[pid])
            print(f"✅ ID {pid} - Color final confirmado: {confirmed_colors[pid]}")
        elif colors:
            most_common_color, _ = Counter(colors).most_common(1)[0]
            player_colors[pid] = assign_team(most_common_color)
            print(f"🟡 ID {pid} - Color más común asumido: {most_common_color}")

    
    
    player_dorsals = {}
    for pid, dorsals in dorsal_history.items():
        if dorsals:
            most_common_dorsal, _ = Counter(dorsals).most_common(1)[0]
            player_dorsals[pid] = most_common_dorsal
            print(f"✅ ID {pid} - Color final confirmado: {most_common_dorsal}")

    for pid, dorsal in confirmed_dorsals.items():
        if pid not in confirmed_colors:
            print(f"🔁 ID {pid} - Sin color confirmado. Buscando en frames de entradas + 5 frames siguientes.")

            entries = frame_data.get(pid, [])
            sampled_entries = [
                entry for i, entry in enumerate(entries)
                if i % 20 == 0 and entry['posicion'] is not None
            ]

            found_color = False
            for entry in sampled_entries:
                x1, y1, x2, y2 = entry['posicion']
                start_frame = entry['frame']

                for offset in range(6):  
                    current_frame = start_frame + offset
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).resize((960,960))

                    detected_color = None
                    if use_model == 'BLIP':
                        detected_color = detect_player_color_with_BLIP(pil_img, pid)
                    elif use_model == 'PaliGemma':
                        detected_color = detect_player_color_with_PaliGemma(pil_img, pid)

                    if detected_color:
                        confirmed_colors[pid] = detected_color
                        print(f"✅ Color recuperado para ID {pid} en frame {current_frame}: {detected_color}")
                        found_color = True
                        break  

                if found_color:
                    break  

  
    
    team_dorsals = defaultdict(set)
    invalid_dorsals = set()

    for pid, dorsal in player_dorsals.items():
        team = player_colors.get(pid)
        if team is None:
            continue
        if dorsal in team_dorsals[team]:
            print(f"❌ Dorsal duplicado: el dorsal {dorsal} ya existe en el equipo {team}")
            invalid_dorsals.add(pid)
        else:
            team_dorsals[team].add(dorsal)

    
    for pid in invalid_dorsals:
        print(f"⚠️ Eliminando dorsal duplicado de ID {pid}")
        player_dorsals[pid] = -1  

    confirmed_dorsals = player_dorsals
    guardar_historial_completo(frame_data, player_colors)
    
    return confirmed_dorsals, player_tracker, player_colors

def guardar_historial_completo(frame_data, player_colors, archivo='historial_detecciones.json'):
    historial = {}
    
    for pid, frames in frame_data.items():
        historial[pid] = {
            'color_confirmado': player_colors.get(pid, ('unknown', 'unknown'))[1],
            'dorsal_confirmado': None,  
            'detalles_por_frame': []
        }
        
        
        for frame in frames:
            frame_entry = {
                'frame': frame['frame'],
                'dorsal': frame['dorsal'],
                'posicion': frame['posicion'],
                'color': frame['color'] if frame['color'] else historial[pid]['color_confirmado']
            }
            historial[pid]['detalles_por_frame'].append(frame_entry)
        
        
        dorsales = [f['dorsal'] for f in frames if f['dorsal'] != -1]
        if dorsales:
            
            for i in range(len(dorsales) - 2):
                if dorsales[i] == dorsales[i+1] == dorsales[i+2]:
                    historial[pid]['dorsal_confirmado'] = dorsales[i]
                    break
            
            
            if historial[pid]['dorsal_confirmado'] is None:
                most_common, count = Counter(dorsales).most_common(1)[0]
                if count >= 3:  
                    historial[pid]['dorsal_confirmado'] = most_common
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"historial_detecciones_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(historial, f, indent=4)
    
    print(f"Historial guardado en {filename}")


