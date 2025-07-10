from video_analysis import analyze_video
from video_writer import annotate_video
from config import use_model

def main():
    input_video = "prueba8.mp4"
    output_video = "BLIP_Soccernet_SNGS-186.mp4"
    dorsals_confirmados, tracker_info, player_colors = analyze_video(input_video, use_model)
    annotate_video(input_video, output_video, dorsals_confirmados, player_colors)

if __name__ == "__main__":
    main()