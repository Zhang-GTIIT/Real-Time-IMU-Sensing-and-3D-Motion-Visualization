#include <Wire.h>

#define MPU_ADDR 0x68
#define REG_SMPLRT_DIV 0x19
#define REG_CONFIG 0x1A
#define REG_GYRO_CONFIG 0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1 0x6B
#define REG_WHO_AM_I 0x75

const unsigned long BAUD_RATE = 500000;
const unsigned long SAMPLE_PERIOD_US = 2000;

int16_t ax, ay, az, gx, gy, gz;
unsigned long next_sample_time = 0;

void writeMPU(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

uint8_t readMPUReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1, (uint8_t)true);
  if (Wire.available()) return Wire.read();
  return 0xFF;
}

bool readMPU6Axis() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;

  uint8_t n = Wire.requestFrom(MPU_ADDR, (uint8_t)14, (uint8_t)true);
  if (n != 14) return false;

  ax = (int16_t)((Wire.read() << 8) | Wire.read());
  ay = (int16_t)((Wire.read() << 8) | Wire.read());
  az = (int16_t)((Wire.read() << 8) | Wire.read());

  Wire.read();
  Wire.read();

  gx = (int16_t)((Wire.read() << 8) | Wire.read());
  gy = (int16_t)((Wire.read() << 8) | Wire.read());
  gz = (int16_t)((Wire.read() << 8) | Wire.read());

  return true;
}

void configureMPU6050() {
  writeMPU(REG_PWR_MGMT_1, 0x00);
  delay(100);
  writeMPU(REG_CONFIG, 0x03);
  writeMPU(REG_SMPLRT_DIV, 0x01);
  writeMPU(REG_GYRO_CONFIG, 0x00);
  writeMPU(REG_ACCEL_CONFIG, 0x00);
}

void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin();
  Wire.setClock(400000);
  delay(500);

  configureMPU6050();

  Serial.print("# WHO_AM_I=0x");
  Serial.println(readMPUReg(REG_WHO_AM_I), HEX);
  Serial.println("# ax,ay,az,gx,gy,gz");

  next_sample_time = micros();
}

void loop() {
  unsigned long now = micros();

  if ((long)(now - next_sample_time) >= 0) {
    next_sample_time += SAMPLE_PERIOD_US;

    if (readMPU6Axis()) {
      Serial.print(ax); Serial.print(',');
      Serial.print(ay); Serial.print(',');
      Serial.print(az); Serial.print(',');
      Serial.print(gx); Serial.print(',');
      Serial.print(gy); Serial.print(',');
      Serial.println(gz);
    }
  }
}