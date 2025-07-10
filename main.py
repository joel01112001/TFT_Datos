from video_analysis import analyze_video
from video_writer import annotate_video
from config import use_model

def main():
    input_video = "prueba.mp4"
    output_video = "resultado.mp4"
    dorsals_confirmados, tracker_info, player_colors = analyze_video(input_video, use_model)
    annotate_video(input_video, output_video, dorsals_confirmados, player_colors)

if __name__ == "__main__":
    main()
