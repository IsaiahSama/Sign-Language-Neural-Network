"""This will test for sign language classification"""

import cv2
from cv2.typing import MatLike
import numpy as np
import onnxruntime as ort

from string import ascii_uppercase

def center_crop(frame: MatLike) -> MatLike:
    """Crops a frame to the center of the image."""
    height, width = frame.shape
    start = abs(height - width) // 2
    
    if height > width:
        return frame[start: start + width]
    else:
        return frame[:, start: start + height]

def main():
    # Constants
    
    index_to_letters = list(ascii_uppercase)
    mean = 0.485 * 255
    std = 0.229 * 255
    
    # Create a runnable session with the exported model.
    ort_session = ort.InferenceSession("sign_language.onnx")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read() # Capture the current frame
        if not ret:
            break

        frame = center_crop(frame) # Crops the camera frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Converts from RGB to grayscale
        
        x = cv2.resize(frame, (28, 28)) # Resizes to 28 x 28
        x = (frame - mean) / std # Normalizes the values
        
        x = x.reshaape((1, 1, 28, 28)).astype(np.float32) # Reshapes to (1, 1, 28, 28)
        y = ort_session.run(None, {'input': x})[0] # Runs the model
        
        index = np.argmax(y, axis=1)
        letter = index_to_letters[index]

        # Display the letter inside the frame, and display it back to the user.
        
        cv2.putText(frame, letter, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()