"""This will test for sign language classification"""

import cv2
import numpy as np
import onnxruntime as ort

from string import ascii_uppercase

def main():
    # Constants
    
    index_to_letters = list(ascii_uppercase)
    mean = 0.485 * 255
    std = 0.229 * 255
    
    # Create a runnable session with the exported model.
    ort_session = ort.InferenceSession("sign_language.onnx")
    

if __name__ == "__main__":
    main()