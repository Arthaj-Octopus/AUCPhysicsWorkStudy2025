int ledPin = 3;      // This is the pin for an LED used to indicate if light intensity is greater than the set threshold
int threshold = 512;  // This is an ADJUSTABLE threshold set to adjust the sensitivity of the signal.

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int sensorValue = analogRead(A0); //This is the Analog pin we are using to read input from
    Serial.println("Covered");
  if (sensorValue > threshold) {
    digitalWrite(ledPin, HIGH);      // LED becomes on when light is detected
    Serial.println("Uncovered");
    delay(300);
  } else {
    digitalWrite(ledPin, LOW);
  }
}