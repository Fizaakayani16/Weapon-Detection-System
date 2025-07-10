import numpy as np
import cv2
import imutils
import datetime
import time
import winsound
import os

def log_detection(timestamp):
    """Log detection events to a file"""
    log_file = 'weapon_detection_log.txt'
    with open(log_file, 'a') as f:
        f.write(f'Weapon detected at {timestamp}\n')

def sound_alarm():
    """Play an alarm sound on detection"""
    frequency = 2500  # Set frequency to 2500 Hertz
    duration = 1000  # Set duration to 1000 ms (1 second)
    winsound.Beep(frequency, duration)

# Initialize the cascade classifier
gun_cascade = cv2.CascadeClassifier('cascade.xml')
camera = cv2.VideoCapture(0)

# Initialize variables
firstFrame = None
gun_exist = False
detection_timestamp = None
alarm_cooldown = 3  # Cooldown period in seconds
last_alarm_time = time.time() - alarm_cooldown

# Create a window
cv2.namedWindow('Weapon Detection System', cv2.WINDOW_NORMAL)

print('Starting Weapon Detection System...')
print('Press "q" to quit the application')

while True:
    ret, frame = camera.read()
    if not ret:
        print('Failed to grab frame')
        break

    # Get current timestamp
    timestamp = datetime.datetime.now()
    text = timestamp.strftime('%A %d %B %Y %I:%M:%S%p')
    
    # Resize and process frame
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect weapons
    weapons = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    # Process detections
    if len(weapons) > 0:
        gun_exist = True
        current_time = time.time()
        
        # Handle detection and alerts
        if current_time - last_alarm_time >= alarm_cooldown:
            detection_timestamp = timestamp
            log_detection(detection_timestamp)
            sound_alarm()
            last_alarm_time = current_time

        # Draw rectangles around detected weapons
        for (x, y, w, h) in weapons:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'WEAPON DETECTED!', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Add timestamp to frame
    cv2.putText(frame, text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Display status
    status = 'WEAPON DETECTED' if gun_exist else 'No Weapons Detected'
    status_color = (0, 0, 255) if gun_exist else (0, 255, 0)
    cv2.putText(frame, f'Status: {status}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # Show the frame
    cv2.imshow('Weapon Detection System', frame)

    # Check for quit command
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Reset gun_exist for next frame
    gun_exist = False

# Cleanup
camera.release()
cv2.destroyAllWindows()

# Final detection status
if gun_exist:
    print('\nWARNING: Weapons were detected during this session!')
    print(f'Last detection at: {detection_timestamp}')
else:
    print('\nNo weapons were detected during this session.')
