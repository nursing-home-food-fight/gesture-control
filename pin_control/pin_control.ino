// This sketch can control any PWM-capable pin based on analog commands from Python.
// It expects a command in the format: "pinNumber,value" (e.g., "9,0.75" or "6,0.5").
// The value should be between 0.0 and 1.0, which will be mapped to 0-255 for analogWrite.

String incomingCommand; // A string to hold the incoming command.

void setup() {
  // Start serial communication.
  Serial.begin(9600);
  
  // Initialize pin 6 as output and set it to LOW initially
  pinMode(6, OUTPUT);
  digitalWrite(6, LOW);
  
  // Send a message to confirm the Arduino is running
  Serial.println("Arduino initialized. Ready to control pins.");
}

void loop() {
  // Check if data is available from the serial port.
  if (Serial.available() > 0) {
    // Read the entire command string until a newline character is received.
    incomingCommand = Serial.readStringUntil('\n');

    // Find the comma that separates the pin number from the value.
    int commaIndex = incomingCommand.indexOf(',');

    // If a comma is found, proceed with parsing the command.
    if (commaIndex > 0) {
      // Extract the pin number part of the string and convert it to an integer.
      String pinString = incomingCommand.substring(0, commaIndex);
      int pinNumber = pinString.toInt();

      // Extract the analog value part of the string and convert it to a float.
      String valueString = incomingCommand.substring(commaIndex + 1);
      float analogValue = valueString.toFloat();
      
      // Ensure value is between 0.0 and 1.0
      analogValue = constrain(analogValue, 0.0, 1.0);
      
      // Map the 0.0-1.0 range to 0-255 for analogWrite
      int pwmValue = int(analogValue * 255);
      
      // Force a minimum PWM value if not zero to ensure LED is visible
      if (analogValue > 0 && pwmValue < 30) {
        pwmValue = 30;  // Set a minimum visible brightness
      }

      // Set the pin as an output BEFORE writing to it.
      pinMode(pinNumber, OUTPUT);
      
      // For full on/off control with more power
      if (analogValue >= 0.95) {
        digitalWrite(pinNumber, HIGH);
        Serial.println("Using digitalWrite HIGH for maximum brightness");
      } 
      else if (analogValue <= 0.05) {
        digitalWrite(pinNumber, LOW);
        Serial.println("Using digitalWrite LOW for off state");
      }
      else {
        // Use exponential mapping for better perceived brightness control
        // Human perception of brightness is roughly logarithmic, so we square the value
        // to make the changes more noticeable at lower brightness levels
        int adjustedPwm = int(255 * (analogValue * analogValue));
        
        // Write the PWM value to the specified pin.
        analogWrite(pinNumber, adjustedPwm);
        
        // Update the pwmValue for the response message
        pwmValue = adjustedPwm;
      }

      // Send a confirmation message back to the Python script.
      Serial.print("Pin ");
      Serial.print(pinNumber);
      Serial.print(" set to analog value ");
      Serial.print(analogValue);
      Serial.print(" (PWM: ");
      Serial.print(pwmValue);
      Serial.println(")");
    }
  }
}
