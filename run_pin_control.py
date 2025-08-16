import serial
import time

# --- Configuration Section ---
ARDUINO_PORT = '/dev/cu.usbmodem2101'
BAUD_RATE = 9600

try:
    # Establish a connection to the Arduino
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=.1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2) # Wait for the connection to establish

    while True:
        # Get user input
        pin = input("Enter pin number to control (e.g., 8) or 'quit' to exit: ")
        if pin.lower() == 'quit':
            break
            
        state = input("Enter state (1 for ON, 0 for OFF): ")

        # Create the command string in the "pin,state" format
        command = f"{pin},{state}\n" # The '\n' is important!

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