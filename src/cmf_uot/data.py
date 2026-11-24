
"""Loaders for the acoustic-camera dataset (see data/README.md)."""
import json
import os
import numpy as np

from .paths import DATA_DIR

def load_camera_detections(file_path='data/camera_detections.json'):
    """
    Loads camera detections from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a camera detection.
              Returns an empty list if the file is not found or is invalid.
    """
    try:
        with open(file_path, 'r') as f:
            detections = json.load(f)
        return detections
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return []

def load_sigs(manip, data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    file_path = os.path.join(data_dir, 'waves', f'mesure_manip{manip}_all.npy')
    try:
        sigs = np.load(file_path)
        return sigs
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None