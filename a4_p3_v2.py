# Alpha Channel Communication

import cv2
import numpy as np
from scipy.fft import fft, fftfreq
import threading
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

MESSAGE = "HI"
DELTA_ALPHA = 0.005            
DISTANCE_CM = 20              

FPS = 30                      
CAMERA_INDEX = 0              
BACKGROUND_IMAGE = "finalFrame.png"

FREQ_BIT_0 = 10               
FREQ_BIT_1 = 15               

# ============================================================================

def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    if len(binary) < 8:
        return ""
    chars = []
    for i in range(0, len(binary) - 7, 8):
        byte = binary[i:i+8]
        try:
            char = chr(int(byte, 2))
            if 32 <= ord(char) <= 126:
                chars.append(char)
            else:
                chars.append('?')
        except:
            chars.append('?')
    return ''.join(chars)

# ============================================================================
# TRANSMITTER
# ============================================================================

def run_transmitter(binary_message):
    background = cv2.imread(BACKGROUND_IMAGE)    
    background = cv2.resize(background, (640, 480))
    
    # BFSK encoding
    alpha_sequence = []
    for bit in binary_message:
        if bit == '0':
            period = int(FPS / FREQ_BIT_0)
            half = period // 2
            alpha_sequence.extend([DELTA_ALPHA] * half + [0] * half)
            alpha_sequence.extend([DELTA_ALPHA] * half + [0] * half)
        else:
            period = int(FPS / FREQ_BIT_1)
            half = period // 2
            alpha_sequence.extend([DELTA_ALPHA] * half + [0] * half)
            alpha_sequence.extend([DELTA_ALPHA] * half + [0] * half)
    
    cv2.namedWindow('TX', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TX', 640, 480)
    
    idx = 0
    while True:
        alpha = alpha_sequence[idx % len(alpha_sequence)]
        frame = (background.astype(float) * (1 - alpha)).astype(np.uint8)
        
        cv2.imshow('TX', frame)
        if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
            break
        idx += 1
    
    cv2.destroyAllWindows()

# ============================================================================
# RECEIVER
# ============================================================================

def run_receiver(binary_message):
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    cv2.namedWindow('RX', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RX', 640, 480)
    
    bits_needed = len(binary_message)
    frames_per_bit = int(FPS / FREQ_BIT_0 * 2)
    buffer_size = bits_needed * frames_per_bit * 3
    
    intensity_buffer = deque(maxlen=buffer_size)
    
    decoded_text = "..."
    last_decode = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray)
        intensity_buffer.append(intensity)
        
        # Decode periodically
        now = time.time()
        if now - last_decode > 0.5 and len(intensity_buffer) >= bits_needed * frames_per_bit:
            decoded_binary = decode(list(intensity_buffer), bits_needed, frames_per_bit)
            decoded_text = binary_to_text(decoded_binary)
            if not decoded_text:
                decoded_text = "..."
            
            # Debug output
            if decoded_binary:
                errors = sum(1 for i in range(min(len(binary_message), len(decoded_binary))) 
                           if binary_message[i] != decoded_binary[i])
                ber = errors / len(binary_message) * 100
                print(f"Decoded: '{decoded_text}' | BER: {ber:.0f}% | Binary: {decoded_binary}")
            
            last_decode = now
        
        # Display decoded message
        display = frame.copy()
        cv2.putText(display, decoded_text, (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
        
        cv2.imshow('RX', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nFinal: '{decoded_text}'")

def decode(intensities, num_bits, frames_per_bit):
    
    if len(intensities) < frames_per_bit * num_bits:
        return ""
    
    signal = intensities[-num_bits * frames_per_bit:]
    decoded = []
    
    for bit_idx in range(num_bits):
        start = bit_idx * frames_per_bit
        end = start + frames_per_bit
        
        if end > len(signal):
            break
        
        window = np.array(signal[start:end])
        window = window - np.mean(window)
        
        fft_vals = np.abs(fft(window))
        freqs = fftfreq(len(window), 1/FPS)
        
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        if len(pos_fft) < 3:
            continue
        
        idx_0 = np.argmin(np.abs(pos_freqs - FREQ_BIT_0))
        idx_1 = np.argmin(np.abs(pos_freqs - FREQ_BIT_1))
        
        power_0 = pos_fft[idx_0]
        power_1 = pos_fft[idx_1]
        
        if power_0 > power_1:
            decoded.append('0')
        else:
            decoded.append('1')
    
    return ''.join(decoded)

# ============================================================================
# MAIN
# ============================================================================

def main():
    
    binary = text_to_binary(MESSAGE)
    
    print(f"\nMessage: '{MESSAGE}'")
    print(f"Binary: {binary} ({len(binary)} bits)")
    print(f"Alpha: {DELTA_ALPHA*100:.1f}%")
    print(f"Distance: {DISTANCE_CM}cm")
    print(f"\nStarting in 3 sec...")
    time.sleep(3)
    
    tx_thread = threading.Thread(target=run_transmitter, args=(binary,), daemon=True)
    tx_thread.start()
    
    time.sleep(1)
    
    try:
        run_receiver(binary)
    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()