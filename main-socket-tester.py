import socket
import time 


# Establish TCP/IP connection with the UR5 CB3 robot
robot_ip = '192.168.1.102'  # IP address of the UR5 CB3 robot
robot_port = 30002  # Default port for UR5 CB3 communication
# Hardcoded position coordinates
position = [-0.0172, 0.4525, 0.1358, 0.6598, -1.161, 1.6353]
urscript_command = f"movej({position}, a=1.2, v=0.25, t=0, r=0)\n"


try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((robot_ip, robot_port))
    print("Connected to the UR5 CB3 robot")
    
    # Send the URScript command to the robot
    sock.send(str.encode(urscript_command))
    print(f"Sent command: {urscript_command}")
    time.sleep(3)
    # Wait for the robot to execute the command
    response = sock.recv(1024).decode()
    print(f"Received response: {response}")
    

finally:
    sock.close()
    print("Disconnected from the UR5 CB3 robot")