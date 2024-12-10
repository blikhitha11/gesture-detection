# Import modules
import os
import sys
import logging
from fastapi import FastAPI, File, UploadFile
from inference import classify_gesture, detect_hand_landmarks
import uvicorn
from pathlib import Path

# Instantiate FastAPI app
app = FastAPI()

# Setup logger for debugging
logging.basicConfig(level=logging.INFO)
logging.info("FastAPI app initialized")

@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Message indicating the API is up and running.
    """
    return {"message": "Gesture Recognition API is up and running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint for gesture recognition.

    Args:
        file (UploadFile): Video or image file containing hand gestures.

    Returns:
        dict: Detected gesture action.
    """
    content = await file.read()
    logging.info("File received for prediction")

    landmarks = detect_hand_landmarks(content)
    if landmarks is None:
        return {"error": "No hand detected"}

    action = classify_gesture(landmarks)
    logging.info(f"Detected action: {action}")
    return {"action": action}

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)