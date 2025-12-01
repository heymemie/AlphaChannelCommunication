"""
Part 3: Screen-Camera Communication - CORRECT ALPHA VALUES

Assignment specs:
- Delta alpha = 0.1  (means 0.1%, not 10%)
- Delta alpha = 0.5  (means 0.5%, not 50%)

So we use:
- DELTA_ALPHA = 0.001  (0.1%)
- DELTA_ALPHA = 0.005  (0.5%)
"""

import cv2
import numpy as np
from scipy.fft import fft, fftfreq
import threading
import time
from collections import deque

# ============================================================================
# CONFIGURATION - CHANGE THESE FOR EACH TEST
# ============================================================================

MESSAGE = "HI"                # Message to send

# Assignment says "delta alpha=0.1 and 0.5"
# This means 0.1% and 0.5%, not 10% and 50%
DELTA_ALPHA = 0.001           # TEST: 0.001 (0.1%) and 0.005 (0.5%)
DISTANCE_CM = 20              # TEST: 20, 40, 80, 120 cm

FPS = 30                      
CAMERA_INDEX = 0              
BACKGROUND_IMAGE = "finalFrame.png"

# BFSK frequencies
FREQ_BIT_0 = 10               
FREQ_BIT_1 = 15               

# ============================================================================
# TEXT/BINARY CONVERSION
# ============================================================================

def text_to_binary(text):
    """Convert text to binary string"""
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def binary_to_text(binary):
    """Convert binary string to text"""
    if len(binary) < 8:
        return ""
    
    num_bytes = len(binary) // 8
    chars = []
    
    for i in range(num_bytes):
        byte = binary[i*8:(i+1)*8]
        try:
            char = chr(int(byte, 2))
            if 32 <= ord(char) <= 126:  # Printable ASCII
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
    """Display image with alpha modulation"""
    
    # Load background
    try:
        background = cv2.imread(BACKGROUND_IMAGE)
        if background is None:
            raise Exception("Could not load image")
    except:
        # Fallback gradient
        background = np.ones((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            background[i, :] = [100 + i//4, 150, 200 - i//4]
    
    background = cv2.resize(background, (640, 480))
    
    # Generate alpha sequence - BFSK encoding
    alpha_sequence = []
    
    for bit in binary_message:
        if bit == '0':
            period_frames = int(FPS / FREQ_BIT_0)
            half_period = period_frames // 2
            for _ in range(2):  # 2 periods per bit
                alpha_sequence.extend([DELTA_ALPHA] * half_period + [0] * half_period)
        else:
            period_frames = int(FPS / FREQ_BIT_1)
            half_period = period_frames // 2
            for _ in range(2):
                alpha_sequence.extend([DELTA_ALPHA] * half_period + [0] * half_period)
    
    cv2.namedWindow('TRANSMITTER', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TRANSMITTER', 640, 480)
    
    frame_idx = 0
    
    while True:
        alpha = alpha_sequence[frame_idx % len(alpha_sequence)]
        
        # Apply alpha dimming
        frame = background.astype(float) * (1 - alpha)
        frame = frame.astype(np.uint8)
        
        cv2.imshow('TRANSMITTER', frame)
        
        if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    cv2.destroyWindow('TRANSMITTER')

# ============================================================================
# RECEIVER - LIVE DECODING
# ============================================================================

def run_receiver(binary_message):
    """Capture and decode continuously"""
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        return
    
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    cv2.namedWindow('RECEIVER', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RECEIVER', 640, 480)
    
    # Buffer setup
    bits_needed = len(binary_message)
    frames_per_bit = int(FPS / FREQ_BIT_0 * 2)
    buffer_size = bits_needed * frames_per_bit * 2
    
    intensity_buffer = deque(maxlen=buffer_size)
    
    # Decode state
    decoded_text = ""
    ber = 0
    data_rate = 0
    last_decode_time = 0
    total_frames = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray)
        intensity_buffer.append(intensity)
        total_frames += 1
        
        # Decode every 0.3 seconds
        current_time = time.time()
        if current_time - last_decode_time > 0.3 and len(intensity_buffer) >= bits_needed * frames_per_bit:
            decoded_binary = decode_sliding_window(
                list(intensity_buffer), 
                bits_needed, 
                frames_per_bit
            )
            decoded_text = binary_to_text(decoded_binary)
            
            # Calculate BER
            if len(decoded_binary) >= len(binary_message):
                errors = sum(1 for i in range(len(binary_message)) 
                           if i < len(decoded_binary) and binary_message[i] != decoded_binary[i])
                ber = errors / len(binary_message)
            
            # Calculate data rate (correct bits per second)
            elapsed = current_time - start_time
            if elapsed > 0:
                correct_bits = len(binary_message) * (1 - ber)
                data_rate = correct_bits / elapsed
            
            last_decode_time = current_time
        
        # Display
        display = frame.copy()
        
        # Dark overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (630, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
        
        # Decoded text (BIG)
        y_pos = 70
        text_color = (0, 255, 0) if ber < 0.2 else (0, 200, 255) if ber < 0.5 else (0, 0, 255)
        cv2.putText(display, decoded_text if decoded_text else "...", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, text_color, 4)
        
        # Metrics
        y_pos += 70
        cv2.putText(display, f"BER: {ber*100:.1f}%  |  Data Rate: {data_rate:.2f} bps", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Config info (show as 0.1% not 10%)
        y_pos += 45
        cv2.putText(display, f"Alpha: {DELTA_ALPHA*100:.1f}%  |  Distance: {DISTANCE_CM}cm", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow('RECEIVER', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyWindow('RECEIVER')
    
    # Print final results for manual recording
    print("\n" + "="*70)
    print("FINAL RESULTS - Record these for your report:")
    print("="*70)
    print(f"Delta Alpha:  {DELTA_ALPHA*100:.1f}%")
    print(f"Distance:     {DISTANCE_CM} cm")
    print(f"Decoded:      '{decoded_text}'")
    print(f"BER:          {ber*100:.1f}%")
    print(f"Data Rate:    {data_rate:.2f} bps")
    print("="*70 + "\n")

def decode_sliding_window(intensities, num_bits, frames_per_bit):
    """Decode message from sliding window"""
    
    if len(intensities) < frames_per_bit * num_bits:
        return ""
    
    # Use most recent data
    signal = intensities[-num_bits * frames_per_bit:]
    decoded_bits = []
    
    for bit_idx in range(num_bits):
        start = bit_idx * frames_per_bit
        end = start + frames_per_bit
        
        if end > len(signal):
            break
        
        window = np.array(signal[start:end])
        window = window - np.mean(window)  # Remove DC
        
        # FFT
        fft_vals = np.abs(fft(window))
        freqs = fftfreq(len(window), 1/FPS)
        
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        if len(pos_fft) < 2:
            continue
        
        # Find power at target frequencies
        idx_0 = np.argmin(np.abs(pos_freqs - FREQ_BIT_0))
        idx_1 = np.argmin(np.abs(pos_freqs - FREQ_BIT_1))
        
        power_0 = pos_fft[idx_0]
        power_1 = pos_fft[idx_1]
        
        # Decode
        if power_0 > power_1:
            decoded_bits.append('0')
        else:
            decoded_bits.append('1')
    
    return ''.join(decoded_bits)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("Part 3: Screen-Camera Communication")
    print("="*70)
    
    binary_message = text_to_binary(MESSAGE)
    
    print(f"\nTEST CONFIGURATION:")
    print(f"  Message:      '{MESSAGE}'")
    print(f"  Delta Alpha:  {DELTA_ALPHA*100:.1f}%  (value: {DELTA_ALPHA})")
    print(f"  Distance:     {DISTANCE_CM} cm")
    print(f"  Binary bits:  {len(binary_message)}")
    
    print(f"\nINSTRUCTIONS:")
    print(f"  1. Position webcam {DISTANCE_CM}cm from transmitter window")
    print(f"  2. Wait for decoded text to appear")
    print(f"  3. When stable, record BER and data rate")
    print(f"  4. Press 'q' to finish")
    print(f"  5. Edit DELTA_ALPHA or DISTANCE_CM at top of script")
    print(f"  6. Run again for next test")
    
    print(f"\nREMINDER - Tests needed for assignment:")
    print(f"  DELTA_ALPHA = 0.001 (0.1%): distances 20, 40, 80, 120 cm  (4 tests)")
    print(f"  DELTA_ALPHA = 0.005 (0.5%): distances 20, 40, 80, 120 cm  (4 tests)")
    print(f"  Total: 8 tests")
    
    if DELTA_ALPHA > 0.01:
        print(f"\n  WARNING: Current DELTA_ALPHA = {DELTA_ALPHA} ({DELTA_ALPHA*100}%)")
        print(f"           This seems too high! Assignment says 0.1 and 0.5")
        print(f"           Should be 0.001 (0.1%) or 0.005 (0.5%)")
    
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Start transmitter
    tx_thread = threading.Thread(target=run_transmitter, args=(binary_message,), daemon=True)
    tx_thread.start()
    
    time.sleep(0.5)
    
    # Run receiver
    try:
        run_receiver(binary_message)
    except KeyboardInterrupt:
        print("\n\nStopped")

if __name__ == "__main__":
    main()