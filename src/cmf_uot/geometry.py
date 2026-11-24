
"""Search grid, microphone array geometry and free-field steering vectors."""
import configparser
import numpy as np

from .paths import CONFIG_FILE, DATA_DIR

def get_grid_from_config(config_path=None):
    """
    Reads grid parameters from a configuration file and returns a 3D grid.

    Args:
        config_path (str): The path to the configuration file (e.g., 'ot.mmcfg').

    Returns:
        numpy.ndarray: A (N, 3) array representing the 3D grid points, or None if an error occurs.
    """
    if config_path is None:
        config_path = CONFIG_FILE
    try:
        cp = configparser.ConfigParser()
        # Check if the file is read successfully
        if not cp.read(config_path):
            print(f"Error: Configuration file not found at {config_path}")
            return None

        # Read grid parameters from the [grid] section and convert to meters
        xg = np.arange(int(cp['grid']['X1']), int(cp['grid']['X2']), int(cp['grid']['DX'])) / 1000
        yg = np.arange(int(cp['grid']['Y1']), int(cp['grid']['Y2']), int(cp['grid']['DY'])) / 1000
        zg = np.arange(int(cp['grid']['Z1']), int(cp['grid']['Z2']), int(cp['grid']['DZ'])) / 1000

        # Create a 3D meshgrid
        Xg, Yg, Zg = np.meshgrid(xg, yg, zg)

        # Flatten and stack the grid points into an (N, 3) array
        Grid = np.vstack([Xg.flatten(), Yg.flatten(), Zg.flatten()]).T

        return Grid,xg, yg, zg

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Error reading config file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def get_array_from_npy(config_path=None, x_position=-3):
    """
    Reads microphone array parameters from a numpy file and returns the array geometry.

    Args:
        config_path (str): The path to the numpy file.
        x_position (int) : The x-coordinate position of the array in meters (default is -3).
    Returns:
        numpy.ndarray: A (M, 3) array representing the microphone positions, or None if an error occurs.
    """
    if config_path is None:
        config_path = DATA_DIR / 'waves' / 'array.npy'
    try:
        array = np.load(config_path)

        Xa = array[:,0]
        Ya = array[:,1]
        Za = array[:,2]

        # Apply transformations 
        Xa_global = Za + x_position
        Ya_global = -Xa + 0.645
        Za_global = Ya
        Array = np.vstack([Xa_global, Ya_global, Za_global]).T

        return Array

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Error reading config file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def freefield(PX, PS, k, normalized=False):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[: ,1] - PS[1]
        dz = PX[:, 2] - PS[2]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T
        dz = PX[:, 2:3] - PS[:, 2:3].T

    d = np.sqrt(dx*dx + dy*dy + dz*dz);
        
    D = np.exp( -1j * k * d) / d
    
    if normalized:
        D = D / np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        
    return D
