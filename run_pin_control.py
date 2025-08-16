import serial
import time

# --- Configuration Section ---
ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600

try:
    # Establish a connection to the Arduino
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=.1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2) # Wait for the connection to establish

    while True:
        # Get user input
        pin = input("Enter PWM-capable pin number (e.g., 3, 5, 6, 9, 10, 11) or 'quit' to exit: ")
        if pin.lower() == 'quit':
            break
            
        value = input("Enter analog value (0.0 to 1.0, where 0.0 is OFF and 1.0 is full intensity): ")
        
        # Validate the input
        try:
            float_value = float(value)
            if float_value < 0.0 or float_value > 1.0:
                print("Warning: Value should be between 0.0 and 1.0. It will be constrained.")
        except ValueError:
            print("Error: Please enter a valid number between 0.0 and 1.0")
            continue

        # Create the command string in the "pin,value" format
        command = f"{pin},{value}\n" # The '\n' is important!

        # Send the command to the Arduino
        arduino.write(command.encode('utf-8'))
        print(f"Sent command: {command.strip()}")
        
        # Wait a moment and read the confirmation from Arduino
        time.sleep(0.1)
        response = arduino.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino response: {response}")
        print("-" * 20)

except serial.SerialException as e:
    print(f"Error: {e}")
finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Serial connection closed.")