// This code runs on the Arduino board.
// It waits for a character ('H' or 'L') from the serial port
// and turns the built-in LED on or off accordingly.


const int LED_PIN = LED_BUILTIN; // Use the built-in LED on the board

void setup() {
  // Start serial communication at 9600 bits per second (baud rate)
  Serial.begin(9600);
  
  // Set the LED pin to be an output
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  // Check if there is any data from the computer waiting to be read
  if (Serial.available() > 0) {
    // Read the incoming character
    char command = Serial.read();

    if (command == 'H') {
      digitalWrite(LED_PIN, HIGH); // Turn the LED on
      Serial.println("LED is ON"); // Send a confirmation message back
    } 
    else if (command == 'L') {
      digitalWrite(LED_PIN, LOW);  // Turn the LED off
      Serial.println("LED is OFF"); // Send a confirmation message back
    }
  }
}