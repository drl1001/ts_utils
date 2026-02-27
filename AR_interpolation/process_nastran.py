'''
Functions to extract GRID node IDs and coordinates from NASTRAN BDF files.
Option to save the NODE IDs and coordinates as numpy arrays.
'''

import numpy as np
from pyNastran.bdf.bdf import BDF


def extract_3D_grid(bdf_path, out_name, save_arrays=False, sort_ids=True, xref=False, punch=False, validate=False):
    """
    Extract GRID node IDs and coordinates from a BDF file
    and return a NumPy array of shape (N,4):

        [node_id, x, y, z]

    Parameters
    ----------
    bdf_path : str
        Path to the BDF file.
    sort_ids : bool, optional
        Sort rows by node ID (default=True).
    xref : bool, optional
        Cross-reference model so coordinates are returned
        in the basic coordinate system (default=True).

    Returns
    -------
    node_array : numpy.ndarray
        Array of shape (N,4):
        column 0 --> node ID
        columns 1-3 --> x, y, z
    """
    model = BDF()
    model.read_bdf(bdf_path, xref=xref, validate=validate, punch=punch)

    n_nodes = len(model.nodes)
    node_array = np.zeros((n_nodes, 4))

    for i, (nid, node) in enumerate(model.nodes.items()):
        node_array[i, 0] = int(nid)
        node_array[i, 1:] = node.get_position()

    if sort_ids:
        node_array = node_array[np.argsort(node_array[:, 0])]
    
    if save_arrays:
        np.save(out_name, node_array)

    return node_array



import numpy as np


def select_root_nodes(nodes_3D: np.ndarray, y_root: float) -> np.ndarray:
    '''
    Extract nodes that are on the root of the wing. Do not interpolate forces from
    aerodynamic mesh onto these root nodes (they are buried within the fuselage in reality).
    These root structural nodes also have 0 displacement.

    Nodes with y coordinate less than y_root are classified as root structural nodes.

    Parameters:
    -----------
    nodes_3D : numpy.ndarray
        Array shape (sn, 4), each row of form (NodeID, x, y, z)

    y_root : float
        Spanwise cutoff value defining wing root

    Returns:
    --------
    wing_nodes : numpy.ndarray
        Structural nodes on the wing surface (y >= y_root)
    root_nodes : numpy.ndarray
        Structural nodes in wing root (y < y_root)
    '''

    if nodes_3D.ndim != 2 or nodes_3D.shape[1] != 4:
        raise ValueError("nodes_3D must have shape (N, 4) -> (NodeID, x, y, z)")

    # Root structural nodes (buried in fuselage)
    root_nodes = nodes_3D[nodes_3D[:, 2] < y_root]

    # Wing surface structural nodes
    wing_nodes = nodes_3D[nodes_3D[:, 2] >= y_root]

    return wing_nodes, root_nodes