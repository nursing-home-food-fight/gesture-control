#include <Servo.h>
Servo myServo; // Create a Servo object
String incomingCommand; // A string to hold the incoming command

// Define servo position limits
const int MIN_POS = 1;    // Minimum servo position (degrees, avoid 0 for reliability)
const int MAX_POS = 89;   // Maximum servo position (degrees, avoid 90 for reliability)
const int INIT_POS = 1;   // Initial position (degrees)

// Timeout handling
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 5000; // 5 seconds timeout

// Serial buffer management
const int MAX_COMMAND_LENGTH = 10;
char serialBuffer[MAX_COMMAND_LENGTH];
int bufferIndex = 0;

void setup() {
  myServo.attach(9); // Attach the servo to digital pin 9
  Serial.begin(9600);
  myServo.write(INIT_POS); // Set to initial position
  Serial.println("Servo control initialized. Send values between 0.0 and 1.0");
  
  // Mark current time as the last command time
  lastCommandTime = millis();
}

void loop() {
  // Check for timeout - if we haven't received a command in a while, keep the servo in the last position
  unsigned long currentTime = millis();
  if (currentTime - lastCommandTime > COMMAND_TIMEOUT) {
    // No commands for a while, just continue (don't reset servo)
    // This prevents the servo from twitching if communication is temporarily lost
  }

  // Process any incoming data efficiently
  while (Serial.available() > 0) {
    // Read one character at a time
    char inChar = (char)Serial.read();
    
    // Check for end of command
    if (inChar == '\n' || inChar == '\r') {
      // Only process if we have content
      if (bufferIndex > 0) {
        // Null-terminate the string
        serialBuffer[bufferIndex] = '\0';
        
        // Parse the command value
        float value = atof(serialBuffer);
        
        // Constrain value to be between 0.0 and 1.0
        value = constrain(value, 0.0, 1.0);
        
        // Calculate servo position (map 0.0-1.0 to servo range)
        // Convert float 0.0-1.0 to servo range 1-89
        // Note: We're inverting the mapping direction (using MAX_POS to MIN_POS)
        // because of how the servo is oriented in your setup
        int servoPosition = map(int(value * 100), 0, 100, MAX_POS, MIN_POS);
        
        // Ensure we stay within safe limits
        servoPosition = constrain(servoPosition, MIN_POS, MAX_POS);
        
        // Set the servo position directly
        myServo.write(servoPosition);
        
        // Log the received value and the servo position (only occasionally to reduce serial traffic)
        if (random(5) == 0) { // Only log 20% of commands to reduce serial traffic
          Serial.print("Received: ");
          Serial.print(value, 2);
          Serial.print(" -> Servo: ");
          Serial.println(servoPosition);
        }
        
        // Reset for next command
        bufferIndex = 0;
        
        // Update the last command time
        lastCommandTime = currentTime;
      }
    }
    // Add character to buffer if not full
    else if (bufferIndex < MAX_COMMAND_LENGTH - 1) {
      serialBuffer[bufferIndex++] = inChar;
    }
  }
  
  // Small delay to prevent CPU hogging
  delay(1);
}