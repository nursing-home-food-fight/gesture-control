# Gesture Control

This repository contains code that uses your webcam to watch gestures and control Arduino boards.

## Development

### Getting Started

1. Install uv for environment management: https://docs.astral.sh/uv/getting-started/installation/

2. Run the following to install dependencies:

```bash
uv venv
uv sync
```

3. Download the hand tracking model

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

4. Run the script to start the program

```bash
uv run main.py
```
