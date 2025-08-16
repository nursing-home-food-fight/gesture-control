// This sketch can control any digital pin based on commands from Python.
// It expects a command in the format: "pinNumber,state" (e.g., "8,1" or "12,0").

String incomingCommand; // A string to hold the incoming command.

void setup() {
  // Start serial communication.
  Serial.begin(9600);
}

void loop() {
  // Check if data is available from the serial port.
  if (Serial.available() > 0) {
    // Read the entire command string until a newline character is received.
    incomingCommand = Serial.readStringUntil('\n');

    // Find the comma that separates the pin number from the state.
    int commaIndex = incomingCommand.indexOf(',');

    // If a comma is found, proceed with parsing the command.
    if (commaIndex > 0) {
      // Extract the pin number part of the string and convert it to an integer.
      String pinString = incomingCommand.substring(0, commaIndex);
      int pinNumber = pinString.toInt();

      // Extract the state part of the string and convert it to an integer.
      String stateString = incomingCommand.substring(commaIndex + 1);
      int state = stateString.toInt();

      // Set the pin as an output BEFORE writing to it.
      pinMode(pinNumber, OUTPUT);
      
      // Write the state (HIGH for 1, LOW for 0) to the specified pin.
      digitalWrite(pinNumber, state);

      // Send a confirmation message back to the Python script.
      Serial.print("Pin ");
      Serial.print(pinNumber);
      Serial.print(" set to state ");
      Serial.println(state);
    }
  }
}
