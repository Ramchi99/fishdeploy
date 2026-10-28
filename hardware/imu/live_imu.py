import time
import board
import busio
import adafruit_bno055

# Initialize the I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize the BNO055 sensor
sensor = adafruit_bno055.BNO055_I2C(i2c)

print("--- BNO055 IMU Live Stream ---")
print("Waiting for sensor calibration... (Move it around in figure-8s if it stays stuck)")

while True:
    # 1. ORIENTATION (Euler Angles: Heading, Roll, Pitch in degrees)
    # Heading (0-360), Roll (-90 to 90), Pitch (-180 to 180)
    heading, roll, pitch = sensor.euler
    
    # 2. LINEAR ACCELERATION (m/s^2)
    # Acceleration with gravity removed. Good for detecting sudden movement/bumps.
    lin_accel_x, lin_accel_y, lin_accel_z = sensor.linear_acceleration
    
    # 3. CALIBRATION STATUS
    # Returns a tuple of (System, Gyro, Accel, Mag) - 0 is uncalibrated, 3 is fully calibrated
    calib_status = sensor.calibration_status

    # Print the data cleanly
    print(f"Heading: {heading:>5.1f} | Roll: {roll:>5.1f} | Pitch: {pitch:>5.1f} | "
          f"Accel X: {lin_accel_x:>5.2f} Y: {lin_accel_y:>5.2f} Z: {lin_accel_z:>5.2f} | "
          f"Calib: {calib_status}")
          
    # Run at roughly 10 Hz
    time.sleep(0.1)
