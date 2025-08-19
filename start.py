from frame_handler import process_frame_with_hands
from video import run_video

def main():
    run_video([process_frame_with_hands])

if __name__ == "__main__":
    main()