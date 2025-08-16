// Simple LED Test Sketch
// This will blink an LED connected to pin 6

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  Serial.println("LED Test Started");
  
  // Initialize pin 6 as output
  pinMode(6, OUTPUT);
}

void loop() {
  // Turn LED on
  digitalWrite(6, HIGH);
  Serial.println("LED ON");
  delay(1000);  // Wait for 1 second
  
  // Turn LED off
  digitalWrite(6, LOW);
  Serial.println("LED OFF");
  delay(1000);  // Wait for 1 second
  
  // Use PWM to set LED to half brightness
  Serial.println("LED 50%");
  analogWrite(6, 128);
  delay(1000);
  
  // Use PWM to set LED to low brightness
  Serial.println("LED 20%");
  analogWrite(6, 50);
  delay(1000);
  
  // Use PWM to set LED to high brightness
  Serial.println("LED 80%");
  analogWrite(6, 200);
  delay(1000);
}
