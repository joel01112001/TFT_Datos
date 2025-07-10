import cv2
import numpy as np
import torch
import json
from config import *


with open("players.json", "r", encoding="utf-8") as f:
    jugadores_db = json.load(f)

def annotate_video(input_path, output_path, confirmed_dorsals, player_colors):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    player_tracker = {}
    next_id = 1

    ball_possession = None
    ball_possession_counter = 0
    BALL_DETECTION_COOLDOWN = 30  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, imgsz=640, half=True, device='0' if torch.cuda.is_available() else 'cpu')[0]

        player_boxes = []
        for box in results.boxes:
            if yolo_model.names[int(box.cls[0])] == 'person':
                player_boxes.append(box)

        
        ball_boxes = [box for box in results.boxes if yolo_model.names[int(box.cls[0])] == 'sports ball']

        if ball_boxes:
            ball = ball_boxes[0]
            x1b, y1b, x2b, y2b = map(int, ball.xyxy[0])
            ball_cx, ball_cy = (x1b + x2b) // 2, (y1b + y2b) // 2

            closest_player_id = None
            min_distance = float('inf')

            for pid, data in player_tracker.items():
                px, py = data['pos']
                distance = np.linalg.norm(np.array([px, py]) - np.array([ball_cx, ball_cy]))
                if distance < min_distance:
                    min_distance = distance
                    closest_player_id = pid

            if closest_player_id and min_distance < 100:  
                ball_possession = closest_player_id
                ball_possession_counter = BALL_DETECTION_COOLDOWN

        
        if ball_possession_counter > 0:
            ball_possession_counter -= 1
        else:
            ball_possession = None

        
        for box in player_boxes:
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

            player_tracker[best_id] = {'pos': (cx, cy)}

            
            if best_id in confirmed_dorsals and best_id in player_colors:
                team, color_name = player_colors[best_id]
                color = COLOR_MAP_RGB.get(color_name, (0, 255, 0))
                dorsal = confirmed_dorsals[best_id]
                name = jugadores_db.get(team.lower(), {}).get(str(dorsal), f"{team.capitalize()} - {dorsal}")
                if team is None:
                    label = f"{dorsal}"
                elif team == "0":
                    label = f"{dorsal}"
                elif team == 'referee':
                    label = "Referee"
                elif team == 'goalkeeper':
                    label = f"Goalkeeper - {name}"
                else:
                    label = f"{name}  {dorsal}"

                cv2.putText(frame, label, (cx - 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                
            elif best_id in player_colors:
                team, color_name = player_colors[best_id]
                color = COLOR_MAP_RGB.get(color_name, (255, 255, 255))  
                label = f"{team.capitalize()}"
                cv2.putText(frame, label, (cx - 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        
        if ball_possession and ball_possession in confirmed_dorsals:
            dorsal = confirmed_dorsals[ball_possession]
            team, color_name = player_colors.get(ball_possession, ("", ""))
            name = jugadores_db.get(team.lower(), {}).get(str(dorsal), f"{team.capitalize()} - {dorsal}")
            label = f"Poseedor del balon: ¨{team}({name})"
            cv2.putText(frame, label, (50, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

       
        

        out.write(frame)

    cap.release()
    out.release()
    print(confirmed_dorsals, player_colors)
    print("✅ Video anotado guardado en:", output_path)
