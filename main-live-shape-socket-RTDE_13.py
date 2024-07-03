from time import sleep
import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import rtde_receive
import rtde_control

print("Environment Ready")

####################################################################
# Load the MelonMasterV1 model
# model = YOLO('yolov8n.pt')  # Replace with the path to the MelonMasterV1 weights file
model = YOLO('RealMelonMasterV1.pt')
model.conf = 0.5  # Set confidence threshold to 0.5
####################################################################
# setup the camera (realsense D415)
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        print("Camera found: ", device_product_line)
        break
if not found_rgb:
    print("The code requires a depth camera with Color sensor")
    exit(0)

# Configure depth and color streams for Intel RealSense D415
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create a window to display the video stream
cv2.namedWindow("2D/3D Video Stream", cv2.WINDOW_NORMAL)
####################################################################
# RTDE configuration
ROBOT_IP = "192.168.1.102"  # IP address of your UR5 CB3 robot
FREQUENCY = 125  # Frequency of data retrieval in Hz

# Establish RTDE connections
rtde_receiver = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_controler = rtde_control.RTDEControlInterface(ROBOT_IP)

####################################################################
# Defining all required global variables
camera_to_tcp_offset_wrt_k1 = (0, 0.06, 0.03, 0, 0, 0)  # Verify the values !!!
safety_distance_tcp_to_watermelon = (0, 0, -0.05, 0, 0, 0)  # Safety distance to keep from the watermelon in meters
radius = None  # The radius of the watermelon, defined globally because i need it outside the main loop
list_of_watermelons_wrt_K1 = []  # List of watermelons detected
drop_off_location_wrt_K0 = [0.110, 0.474, 0.608, 0.005, 2.223, 2.223]  # The drop-off location in the K0 coordinate system
####################################################################
# Define all the functions, speaking function names

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Converts an axis-angle rotation to a 3x3 rotation matrix using Rodrigues' rotation formula.
    Parameters:
        axis_angle (tuple or list): Axis-angle rotation vector (rx, ry, rz)
    Returns:
        np.array: 3x3 rotation matrix
    """
    if np.all(np.array(axis_angle) == 0):
        return np.identity(3)
    
    axis = np.array(axis_angle) / np.linalg.norm(axis_angle)
    angle = np.linalg.norm(axis_angle)
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle
    x, y, z = axis
    
    rotation_matrix = np.array([
        [cos_angle + x**2 * one_minus_cos, x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
        [y*x*one_minus_cos + z*sin_angle, cos_angle + y**2 * one_minus_cos, y*z*one_minus_cos - x*sin_angle],
        [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z**2 * one_minus_cos]
    ])
    
    return rotation_matrix

def pose_to_transformation_matrix(pose):
    """
    Converts a 6D pose vector (position + axis-angle rotation) to a 4x4 homogeneous transformation matrix.
    Parameters:
        pose (tuple or list): 6D pose vector (x, y, z, rx, ry, rz)
    Returns:
        np.array: 4x4 homogeneous transformation matrix
    """
    position = pose[:3]
    axis_angle = pose[3:]

    rotation_matrix = axis_angle_to_rotation_matrix(axis_angle)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix

def transformation_matrix_to_pose(matrix):
    """
    Converts a 4x4 homogeneous transformation matrix to a 6D pose vector (position + axis-angle rotation).
    Parameters:
        matrix (np.array): 4x4 homogeneous transformation matrix
    Returns:
        np.array: 6D pose vector (x, y, z, rx, ry, rz)
    """
    position = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    axis_angle = rotation_matrix_to_axis_angle(rotation_matrix)
    pose = np.concatenate([position, axis_angle])
    
    return pose

def rotation_matrix_to_axis_angle(rotation_matrix):
    """
    Converts a 3x3 rotation matrix to an axis-angle rotation vector.
    Parameters:
        rotation_matrix (np.array): 3x3 rotation matrix
    Returns:
        np.array: Axis-angle rotation vector (rx, ry, rz)
    """
    assert np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(3)), "Input is not a valid rotation matrix"
    
    angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    
    if angle == 0:
        return np.array([0, 0, 0])
    elif angle == np.pi:
        x = np.sqrt((rotation_matrix[0, 0] + 1) / 2)
        y = np.sqrt((rotation_matrix[1, 1] + 1) / 2) * np.sign(rotation_matrix[0, 1])
        z = np.sqrt((rotation_matrix[2, 2] + 1) / 2) * np.sign(rotation_matrix[0, 2])
        return np.array([x, y, z]) * angle
    else:
        rx = rotation_matrix[2, 1] - rotation_matrix[1, 2]
        ry = rotation_matrix[0, 2] - rotation_matrix[2, 0]
        rz = rotation_matrix[1, 0] - rotation_matrix[0, 1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        return axis * angle

def detect_watermelon(model, frame, depth_image, intrinsics):
    """
    Detects watermelons in the given frame using the model and returns the watermelon center and radius.
    """
    results = model(frame)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            if conf > 0.5:  # Adjust the confidence threshold as needed
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls]

                # Extract the depth values within the bounding box
                bbox_depth = depth_image[y1:y2, x1:x2]

                # Create a mask to ignore zero depth values
                mask = bbox_depth > 0

                # Convert depth from image plane to camera plane
                depth_points = np.zeros((bbox_depth.shape[0], bbox_depth.shape[1], 3))
                x_indices, y_indices = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
                depth_points[mask, 0] = (x_indices[mask] - intrinsics.ppx) / intrinsics.fx
                depth_points[mask, 1] = (y_indices[mask] - intrinsics.ppy) / intrinsics.fy
                depth_points[mask, 2] = bbox_depth[mask] / 1000.0  # Convert to meters

                # Reshape depth points to a 2D array
                points = depth_points[mask].reshape(-1, 3)

                # Perform DBSCAN clustering to segment the watermelon points
                dbscan = DBSCAN(eps=0.05, min_samples=10)
                labels = dbscan.fit_predict(points)

                # Find the largest cluster (assumed to be the watermelon)
                unique_labels, counts = np.unique(labels, return_counts=True)
                if len(counts) > 1:
                    largest_cluster_label = unique_labels[np.argmax(counts[1:]) + 1]
                    watermelon_points = points[labels == largest_cluster_label]
                    if len(watermelon_points) > 0:
                        # Fit an ellipsoid to the watermelon points
                        hull = ConvexHull(watermelon_points)
                        watermelon_center = np.mean(watermelon_points[hull.vertices], axis=0)
                        distances = np.linalg.norm(watermelon_points[hull.vertices] - watermelon_center, axis=1)
                        radius = np.max(distances)
                        
                        return watermelon_center, radius

    return None, None

####################################################################
# Actual execution of the code
try:
    print("test point -2 reached")
    watermelon_detected = False
    while not watermelon_detected:
        print("test point -1 reached")
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)  # alpha=0.03 is a scaling factor, can be adjusted

        # Perform watermelon detection
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        watermelon_center, radius = detect_watermelon(model, color_image, depth_image, intrinsics)
        if watermelon_center is not None:
            list_of_watermelons_wrt_K1.append(watermelon_center)
            print(f"Watermelon detected at location (wrt K1): {watermelon_center} with radius: {radius:.2f} meters")
            watermelon_detected = True

        # Resize the depth colormap to match the color image size
        depth_colormap_resized = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))

        # Concatenate the color image and depth colormap horizontally
        combined_image = np.concatenate((color_image, depth_colormap_resized), axis=1)
        cv2.imshow("2D/3D Video Stream (adjust alpha for contrast change)", combined_image)
        cv2.waitKey(10)  # Add a small delay (in milliseconds)

        if not watermelon_detected:
            print("test point 4 reached")
            for i in range(4):
                # search_pose = [90, -70, -150, 3, i*45, 180] # Observation position; Cups on the Ground
                search_pose = [90, -70, -150, 13, i * 45, 180]  # Observation position; cups on the box
                # search_pose = [90, -90, -90, 0, i*45, 180] # Home position
                # Convert the joint angles from degrees to radians
                joint_angles_rad_search_pose = np.deg2rad(search_pose)
                rtde_controler.moveJ(joint_angles_rad_search_pose, 0.3, 0.3)
                print("test point 4/i=", i, "reached")

                # Check for watermelon presence after each movement
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                watermelon_center, radius = detect_watermelon(model, color_image, depth_image, intrinsics)
                if watermelon_center is not None:
                    list_of_watermelons_wrt_K1.append(watermelon_center)
                    print(f"Watermelon detected at location (wrt K1): {watermelon_center} with radius: {radius:.2f} meters")
                    watermelon_detected = True
                    break

            if watermelon_detected:
                break

    # Find the closest watermelon
    if watermelon_detected:
        print("test point 3 reached")
        tcp_position = rtde_receiver.getActualTCPPose()
        robot_htm = pose_to_transformation_matrix(tcp_position)  # robot_htm is the homogeneous transformation matrix of the robot

        # Find the closest watermelon in the K1 coordinate system
        distances = [np.linalg.norm(watermelon_loc - tcp_position[:3]) for watermelon_loc in list_of_watermelons_wrt_K1]
        closest_watermelon_index = np.argmin(distances)
        closest_watermelon_loc = list_of_watermelons_wrt_K1[closest_watermelon_index]

        tcp_offset_pose = (closest_watermelon_loc + camera_to_tcp_offset_wrt_k1[:3])
        tcp_offset_htm = pose_to_transformation_matrix(tcp_offset_pose)
        combined_htm = np.dot(robot_htm, tcp_offset_htm)
        new_tcp_position_K0 = transformation_matrix_to_pose(combined_htm)

        # Move the robot to the closest watermelon location
        new_temp_tcp_pose_K0 = list(new_tcp_position_K0[:3]) + list(tcp_position[3:])
        rtde_controler.moveL(new_temp_tcp_pose_K0, 0.1, 0.2)
        print("test point 3.1 reached")
        ######################################
        # Insert the drop-off sequence here
        drop_off_pose_K0 = drop_off_location_wrt_K0
        rtde_controler.moveL(drop_off_pose_K0, 0.1, 0.2)
        ######################################
        
except KeyboardInterrupt:
    # Stop file recording and disconnect on keyboard interrupt
    print("test point 5 reached")
    pass

finally:
    print("Number of watermelons: ", len(list_of_watermelons_wrt_K1))  # Is this the correct way to count the number of watermelons? maybe .shape?
    rtde_receiver.disconnect()
    rtde_controler.disconnect()
    print("Disconnected from the UR5 CB3 robot")
    sleep(3)
    cv2.destroyAllWindows()
    print("Window closed")