"""CMF-UOT: sensor array and camera fusion via unbalanced optimal transport.

Reference implementation of

    I. Jaouedi, G. Chardon, J. Picheral, "On Sensor Array and Camera Fusion via
    Unbalanced Optimal Transport", submitted to Signal Processing (Elsevier).

Modules
-------
geometry    search grid, microphone array, free-field steering vectors
covariance  sample spatial covariance at one frequency bin
camera      pixel-to-world mapping and camera cost matrix C
bounds      Fisher information of the camera measurement (CRB)
data        loaders for the acoustic-camera dataset
solvers     CMF and the two CMF-UOT variants (greedy coordinate descent)
paths       repository, data and results folders
"""

__version__ = "1.0.0"
