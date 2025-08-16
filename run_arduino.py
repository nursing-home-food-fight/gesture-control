# This script runs on your computer.
# It connects to the Arduino and sends commands to control the LED.

import serial
import time

# --- IMPORTANT ---
# The port has been updated as you requested.
ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600

try:
    # Establish a connection to the Arduino
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=.1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    
    # Allow time for the connection to initialize and for the Arduino to reset
    time.sleep(2) 

    # --- Send commands ---
    print("Sending 'H' to turn LED ON...")
    arduino.write(b'H') # Send the byte for 'H'
    time.sleep(2)      # Wait for 2 seconds

    print("Sending 'L' to turn LED OFF...")
    arduino.write(b'L') # Send the byte for 'L'
    time.sleep(1)

    # Read and print any response from Arduino
    responses = arduino.readlines()
    for res in responses:
        print(f"Arduino response: {res.decode('utf-8').strip()}")

except serial.SerialException as e:
    print(f"Error: Could not connect to the Arduino. {e}")
    print("Please check the port name and ensure the Arduino is connected.")

finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Serial connection closed.")