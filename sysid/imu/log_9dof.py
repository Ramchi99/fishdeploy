import time
import board
import busio
import adafruit_bno055
import csv
import datetime

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

# Explicitly set to 9DOF mode (Magnetometer is ON)
sensor.mode = adafruit_bno055.NDOF_MODE

filename = f"data/imu_9DOF_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
headers = [
    "Timestamp_s", "Calib_Sys", "Calib_Gyro", "Calib_Accel", "Calib_Mag",
    "Raw_Accel_X", "Raw_Accel_Y", "Raw_Accel_Z",
    "Raw_Gyro_X", "Raw_Gyro_Y", "Raw_Gyro_Z",
    "Raw_Mag_X", "Raw_Mag_Y", "Raw_Mag_Z",
    "Lin_Accel_X", "Lin_Accel_Y", "Lin_Accel_Z",
    "Quat_W", "Quat_X", "Quat_Y", "Quat_Z",
    "Euler_Yaw", "Euler_Roll", "Euler_Pitch"
]

print("--- 9DOF (NDOF) IMU Logger ---")
print("1. CALIBRATION PHASE: Move the sensor in figure-8s.")

while True:
    sys, gyro, accel, mag = sensor.calibration_status
    print(f"Calibration -> Sys: {sys}, Gyro: {gyro}, Accel: {accel}, Mag: {mag}", end="\r")
    
    # Wait for ALL 4 to calibrate
    if sys == 3 and gyro == 3 and accel == 3 and mag == 3:
        print("\n\nSensor fully calibrated for 9DOF!")
        break
    time.sleep(0.1)

print("2. LOGGING PHASE: Tape it down and DO NOT TOUCH IT.")
time.sleep(3)

start_time = time.time()
try:
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        
        log_count = 0
        while True:
            current_time = time.time() - start_time
            sys, gyro, accel, mag = sensor.calibration_status
            
            a_raw = sensor.acceleration
            g_raw = sensor.gyro
            m_raw = sensor.magnetic
            a_lin = sensor.linear_acceleration
            quat = sensor.quaternion
            euler = sensor.euler

            if all(v is not None for v in [a_raw, g_raw, m_raw, a_lin, quat, euler]):
                writer.writerow([
                    round(current_time, 4), sys, gyro, accel, mag,
                    a_raw[0], a_raw[1], a_raw[2],
                    g_raw[0], g_raw[1], g_raw[2],
                    m_raw[0], m_raw[1], m_raw[2],
                    a_lin[0], a_lin[1], a_lin[2],
                    quat[0], quat[1], quat[2], quat[3],
                    euler[0], euler[1], euler[2]
                ])
                log_count += 1
                if log_count % 500 == 0:
                    print(f"Logged {log_count} rows... (Time: {current_time:.1f}s)")
            
            time.sleep(0.01)

except KeyboardInterrupt:
    print(f"\nSaved {log_count} rows to {filename}")
