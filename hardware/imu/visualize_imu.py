import pygame
import math
import board
import busio
import adafruit_bno055

# 1. Initialize I2C and the BNO055 Sensor
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

# 2. Initialize Pygame Window
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("BNO055 3D Orientation Visualizer")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 32)

# 3. Define the 3D shape (A rectangular block representing your sensor/fish)
vertices = [
    [-1.5, -0.5, -2.5], [1.5, -0.5, -2.5], [1.5, 0.5, -2.5], [-1.5, 0.5, -2.5], # Front face
    [-1.5, -0.5,  2.5], [1.5, -0.5,  2.5], [1.5, 0.5,  2.5], [-1.5, 0.5,  2.5]  # Back face
]

# The lines connecting the corners
edges = [
    (0,1), (1,2), (2,3), (3,0), # Front
    (4,5), (5,6), (6,7), (7,4), # Back
    (0,4), (1,5), (2,6), (3,7)  # Connecting sides
]

def rotate_3d(point, pitch, roll, yaw):
    x, y, z = point
    
    # X-axis rotation (Pitch)
    xy = y * math.cos(pitch) - z * math.sin(pitch)
    xz = y * math.sin(pitch) + z * math.cos(pitch)
    y, z = xy, xz
    
    # Y-axis rotation (Yaw/Heading)
    xx = x * math.cos(yaw) + z * math.sin(yaw)
    xz = -x * math.sin(yaw) + z * math.cos(yaw)
    x, z = xx, xz
    
    # Z-axis rotation (Roll)
    xx = x * math.cos(roll) - y * math.sin(roll)
    xy = x * math.sin(roll) + y * math.cos(roll)
    x, y = xx, xy
    
    return [x, y, z]

print("Starting 3D Visualizer...")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30)) # Dark gray background

    try:
        # Pull live Euler angles from the sensor
        euler = sensor.euler
        if euler is not None and None not in euler:
            yaw_deg, roll_deg, pitch_deg = euler
            
            # Convert degrees to radians for the math (Inverting yaw so it spins intuitively on screen)
            yaw = math.radians(-yaw_deg)
            roll = math.radians(roll_deg)
            pitch = math.radians(pitch_deg)

            projected_vertices = []
            for v in vertices:
                # 1. Scale the block up to look nice on screen
                scaled_v = [v[0]*60, v[1]*60, v[2]*60]
                
                # 2. Apply the physical rotations
                rotated_v = rotate_3d(scaled_v, pitch, roll, yaw)
                
                # 3. Add 3D perspective (things further away look smaller)
                distance = 400
                z_depth = distance / (distance - rotated_v[2]) if (distance - rotated_v[2]) != 0 else 1
                
                x_proj = int(rotated_v[0] * z_depth + width / 2)
                y_proj = int(rotated_v[1] * z_depth + height / 2)
                projected_vertices.append((x_proj, y_proj))

            # Draw the wireframe edges
            for edge in edges:
                pygame.draw.line(screen, (0, 255, 200), projected_vertices[edge[0]], projected_vertices[edge[1]], 3)
                
            # Draw a red highlight over the "Front" face so you know which way it's pointing
            pygame.draw.polygon(screen, (255, 100, 100), 
                                [projected_vertices[0], projected_vertices[1], projected_vertices[2], projected_vertices[3]], 
                                3)

            # Draw Text UI
            text = font.render(f"Yaw: {yaw_deg:.1f}  |  Roll: {roll_deg:.1f}  |  Pitch: {pitch_deg:.1f}", True, (255, 255, 255))
            screen.blit(text, (20, 20))
            
            calib = sensor.calibration_status
            calib_text = font.render(f"Calibration (Sys, Gyro, Accel, Mag): {calib}", True, (150, 150, 150))
            screen.blit(calib_text, (20, 60))

    except Exception as e:
        # I2C can occasionally drop a frame. We just ignore it and keep drawing.
        pass 

    pygame.display.flip()
    clock.tick(60) # Lock at 60 FPS

pygame.quit()
