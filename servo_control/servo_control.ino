#include <Servo.h>
Servo myServo; // Create a Servo object
String incomingCommand; // A string to hold the incoming command

void setup() {
  myServo.attach(9); // Attach the servo to digital pin 9
  Serial.begin(9600);
  myServo.write(90); // Set to initial position
  Serial.println("Servo control initialized. Send value > 0.5 to activate.");
}

void loop() {
  // Check if data is available from the serial port
  if (Serial.available() > 0) {
    // Read the command string until newline character
    incomingCommand = Serial.readStringUntil('\n');
    
    // Parse the command value
    float value = incomingCommand.toFloat();
    
    // Only activate the servo sequence if the value is > 0.5
    if (value > 0.5) {
      runServoSequence();
      Serial.println("Servo sequence completed");
    } else {
      Serial.print("Value ");
      Serial.print(value);
      Serial.println(" is below threshold (0.5). No action taken.");
    }
  }
}

// Function to run the chopping servo sequence
void runServoSequence() {
  int fastDelay = 5;    // Fast movement delay (going down) - was 20
  int slowDelay = 20;   // Slow movement delay (going up) - was 5
  int myInitPos = 90;   // Initial position
  int newPos = myInitPos; // Set new position to initial position
  
  Serial.println("Starting servo sequence...");
  myServo.write(myInitPos);   // Move servo to initial position
  delay(100);
  
  // First movement - going down FAST
  for (int i = 0; i < myInitPos; i++) {
    newPos = myInitPos - i;
    myServo.write(newPos);
    delay(fastDelay); // Fast movement
  }
  
  // Second movement - going up SLOW
  for (int i = 0; i < myInitPos; i++) {
    newPos = i;
    myServo.write(newPos);
    delay(slowDelay); // Slow movement
  }
  
  // Return to initial position
  myServo.write(myInitPos);
  delay(100);
}