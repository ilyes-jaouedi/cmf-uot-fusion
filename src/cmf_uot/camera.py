"""Camera model: pixel-to-world mapping and the angular cost matrix C."""
import numpy as np

camera_position = (7, -300, 90)


def pixel_to_world(u, v, A_matrix, y_fixed=0):
    """
    Estimates 3D world coordinates (x, y, z) from 2D pixel coordinates (u, v)
    using a pre-calculated linear least-squares transformation matrix.

    Args:
        u (float): The u-coordinate of the pixel.
        v (float): The v-coordinate of the pixel.
        A_matrix (np.ndarray): The 2x3 transformation matrix obtained via least-squares.
        y_fixed (float): The fixed y-coordinate, which is assumed to be 0.

    Returns:
        tuple: The estimated world coordinates (x, y, z).
    """
    if A_matrix is None:
        return None, None, None

    # Create the input vector for the new pixel coordinate
    new_U = np.array([[u], [v], [1]])

    # Apply the linear transformation
    new_X = A_matrix @ new_U
    
    # The result is a 2x1 vector [x, z]
    x_est = new_X[0, 0]
    z_est = new_X[1, 0]

    return x_est, y_fixed, z_est

def compute_C(grid_points, detections,A_loaded):
    """
    Computes the cost matrix C based on the angular deviation between the direction
    vectors of camera detections and acoustic grid points.

    Args:
        grid_points (np.ndarray): An (M, 3) array of 3D grid points.
        detections (list): A list of dictionaries, where each dictionary represents a camera detection.
        refs (list): A list of reference points for pixel_to_world conversion.
        camera_position (tuple): The (x, y, z) coordinates of the camera.

    Returns:
        np.ndarray: An (M, N) cost matrix, where M is the number of grid points and N is the number of detections.
    """
    num_grid_points = grid_points.shape[0]
    num_detections = len(detections)
    cost_matrix = np.zeros((num_grid_points, num_detections))

    # Convert pixel coordinates to world coordinates for all detections
    detection_points = np.array([pixel_to_world(det['u'], det['v'],A_loaded) for det in detections])

    # Compute direction vectors from the camera to the detection points
    camera_pos = np.array(camera_position)
    detection_vectors = detection_points - camera_pos
    detection_vectors = detection_vectors / 100 

    # Normalize grid points and detection vectors to get direction vectors
    grid_vectors = grid_points - (camera_pos/100)
    grid_norms = np.linalg.norm(grid_vectors, axis=1, keepdims=True)
    detection_norms = np.linalg.norm(detection_vectors, axis=1, keepdims=True)

    # Avoid division by zero for points at the origin
    grid_norms[grid_norms == 0] = 1
    detection_norms[detection_norms == 0] = 1

    grid_unit_vectors = grid_vectors / grid_norms
    detection_unit_vectors = detection_vectors / detection_norms

    # Compute the dot product between every grid point and every detection
    dot_product_matrix = np.dot(grid_unit_vectors, detection_unit_vectors.T)

    # Clip values to the valid range for arccos to avoid numerical errors
    dot_product_matrix = np.clip(dot_product_matrix, -1.0, 1.0)

    # Compute the angular distance
    cost_matrix = np.arccos(dot_product_matrix)

    return cost_matrix