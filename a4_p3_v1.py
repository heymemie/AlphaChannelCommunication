# Transmitter
import numpy as np
import cv2
from PIL import Image

# Load sample img
background = Image.open('finalFrame.png').convert('RGBA')

# Create black overlay to modulate alpha on
overlay = Image.new('RGBA', background.size, (0, 0, 0, 0)) 

# encoding function, bit 0 20Hz, bit 1 30Hz
def encode_bit(bit, frame_rate=60):
    if bit == 0:
        # 20 Hz: period = 0.05s = 3 frames at 60 FPS
        pattern = [0.05, 0, 0.05, 0, 0.05, 0]
    else:
        # 30 Hz: period = 0.033s = 2 frames at 60 FPS
        pattern = [0.05, 0, 0.05, 0]
        pattern = [0.05, 0, 0.05, 0, 0.05, 0]
    
    return pattern

def generate_alpha_sequence(message, delta_alpha=0.05):

    alpha_sequence = []
    
    for bit in message:
        if bit == '0':
            # 20 Hz oscillation: 3 frames high, 3 frames low
            alpha_sequence.extend([delta_alpha] * 3 + [0] * 3)
        else:  # bit == '1'
            # 30 Hz oscillation: 2 frames high, 2 frames low, repeat
            alpha_sequence.extend([delta_alpha] * 2 + [0] * 2 + [delta_alpha] * 2 + [0] * 2)
    
    return alpha_sequence

def display_transmission(background_img, message, delta_alpha=0.05):
    alpha_sequence = generate_alpha_sequence(message, delta_alpha)
    
    cv2.namedWindow('Transmitter', cv2.WINDOW_NORMAL)
    
    for alpha_value in alpha_sequence:
        # Create overlay with current alpha
        overlay = np.zeros((*background_img.shape[:2], 4), dtype=np.uint8)
        overlay[:, :, 3] = int(alpha_value * 255)  # Set alpha channel
        
        # Blend with background
        display_frame = background_img.copy()
        
        # Simple alpha blending
        if alpha_value > 0:
            display_frame = (display_frame * (1 - alpha_value)).astype(np.uint8)
        
        cv2.imshow('Transmitter', display_frame)
        cv2.waitKey(int(1000/60))  # 60 FPS
    
    cv2.destroyAllWindows()

