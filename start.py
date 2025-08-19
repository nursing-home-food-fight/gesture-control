from numpy import ndarray
from video import run_video

def handle_frame_test(frame: ndarray):
    # Process the frame (this is just a placeholder)
    print("Processing frame:", frame.shape)

def main():
    print("Hello, World!")
    run_video([handle_frame_test])

if __name__ == "__main__":
    main()