"""
PVDataToNumpy.py

Utilities for converting ParaView pipeline proxy objects into NumPy arrays.

This module provides helper functions to:

- Fetch VTK datasets from ParaView pipeline proxies
- Automatically traverse vtkMultiBlockDataSet structures
- Extract PointData and CellData arrays
- Optionally extract mesh point coordinates
- Return everything as NumPy arrays for downstream analysis

Designed for CFD post-processing workflows using pvpython.
"""

import numpy as np
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy


# -----------------------------------------------------------------------------
# Internal: Extract arrays from a concrete VTK dataset
# -----------------------------------------------------------------------------
def _extract_from_dataset(dataset, data_arrays, points=False):
    """
    Extract PointData, CellData, and optionally point coordinates
    from a concrete VTK dataset (e.g. vtkPolyData, vtkUnstructuredGrid).

    This function operates on a *leaf* dataset (i.e. not a multiblock
    container) and appends extracted NumPy arrays to the provided
    dictionary.

    Parameters
    ----------
    dataset : vtkDataSet
        A concrete VTK dataset such as vtkPolyData or
        vtkUnstructuredGrid.

    data_arrays : dict[str, list[np.ndarray]]
        Dictionary used to collect arrays. Each key corresponds to
        an array name, and each value is a list of NumPy arrays.
        Multiple entries occur when processing multiblock datasets.

    points : bool, optional
        If True, extract mesh point coordinates and store them
        under the key "Points".

    Notes
    -----
    - PointData arrays are extracted via dataset.GetPointData().
    - CellData arrays are extracted via dataset.GetCellData().
    - Arrays are converted using vtk_to_numpy for efficient
      zero-copy conversion where possible.
    - Arrays are appended to lists and concatenated later by
      the public API.
    """

    # -----------------------
    # Point Data
    # -----------------------
    pdat = dataset.GetPointData()
    if pdat:
        for i in range(pdat.GetNumberOfArrays()):
            name = pdat.GetArrayName(i)
            vtk_arr = pdat.GetArray(i)
            if vtk_arr:
                arr = vtk_to_numpy(vtk_arr)
                data_arrays.setdefault(name, []).append(arr)

    # -----------------------
    # Cell Data
    # -----------------------
    cdat = dataset.GetCellData()
    if cdat:
        for i in range(cdat.GetNumberOfArrays()):
            name = cdat.GetArrayName(i)
            vtk_arr = cdat.GetArray(i)
            if vtk_arr:
                arr = vtk_to_numpy(vtk_arr)
                data_arrays.setdefault(name, []).append(arr)

    # -----------------------
    # Points
    # -----------------------
    if points and dataset.GetPoints():
        vtk_pts = dataset.GetPoints().GetData()
        pts = vtk_to_numpy(vtk_pts)
        data_arrays.setdefault("Points", []).append(pts)


# -----------------------------------------------------------------------------
# Internal: Recursively traverse multiblock dataset
# -----------------------------------------------------------------------------
def _traverse_multiblock(block, data_arrays, points=False):
    """
    Recursively traverse a vtkMultiBlockDataSet until leaf datasets
    are reached, extracting arrays from each leaf.

    Parameters
    ----------
    block : vtkDataObject
        Either a vtkMultiBlockDataSet or a concrete VTK dataset.

    data_arrays : dict[str, list[np.ndarray]]
        Dictionary used to collect extracted arrays.

    points : bool, optional
        If True, point coordinates are extracted from each leaf dataset.

    Notes
    -----
    ParaView readers frequently produce vtkMultiBlockDataSet objects,
    especially for structured CFD outputs (e.g. multiple domains,
    boundary patches, or zones).

    This function ensures that all nested blocks are visited and
    processed correctly.
    """

    if block.IsA("vtkMultiBlockDataSet"):
        for i in range(block.GetNumberOfBlocks()):
            sub_block = block.GetBlock(i)
            if sub_block:
                _traverse_multiblock(sub_block, data_arrays, points)
    else:
        _extract_from_dataset(block, data_arrays, points)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def PVDataToNumpy(data, points=False):
    """
    Convert a ParaView pipeline proxy into a dictionary of NumPy arrays.

    This is the main public function of the module. It:

    1. Fetches the VTK dataset from the ParaView server
    2. Traverses multiblock structures automatically
    3. Extracts PointData and CellData arrays
    4. Optionally extracts mesh coordinates
    5. Concatenates arrays across blocks

    Parameters
    ----------
    data : paraview.servermanager.Proxy
        A ParaView pipeline object (e.g. Calculator, ExtractSurface,
        CellDatatoPointData, reader, etc.).

        IMPORTANT: This must be a ParaView proxy, not an already
        fetched VTK dataset.

    points : bool, optional, default=False
        If True, include mesh point coordinates under the key "Points".
        The returned array has shape (N_points, 3).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping array names to NumPy arrays.

        - Scalar arrays ? shape (N,)
        - Vector arrays ? shape (N, n_components)
        - Points (if requested) ? shape (N, 3)

    Examples
    --------
    >>> pv_data = PVDataToNumpy(p_force, points=True)
    >>> pv_data["p_force"].shape
    (N, 3)

    >>> total_force = np.sum(pv_data["p_force"], axis=0)

    Notes
    -----
    - Arrays from multiple blocks are concatenated along axis=0.
    - Both PointData and CellData arrays are extracted.
    - If arrays share the same name across blocks, they are merged.
    - This function is intended for pvpython workflows.
    """

    # Fetch dataset from ParaView server
    vtk_data = sm.Fetch(data)

    data_arrays = {}
    _traverse_multiblock(vtk_data, data_arrays, points)

    # Concatenate arrays from multiple blocks
    for key in data_arrays:
        data_arrays[key] = np.concatenate(data_arrays[key], axis=0)

    return data_arrays


# -----------------------------------------------------------------------------
# Optional helper: directly save arrays
# -----------------------------------------------------------------------------
def SaveArrays(data_dict, prefix="pvdata"):
    """
    Save all NumPy arrays in a dictionary to disk as .npy files.

    Parameters
    ----------
    data_dict : dict[str, np.ndarray]
        Dictionary returned by PVDataToNumpy().

    prefix : str, optional, default="pvdata"
        Prefix for output filenames.

        Files will be saved as:
            {prefix}_{array_name}.npy

    Examples
    --------
    >>> pv_data = PVDataToNumpy(p_force, points=True)
    >>> SaveArrays(pv_data, prefix="wing")

    This produces:
        wing_p_force.npy
        wing_Points.npy
        wing_Area.npy
        etc.

    Notes
    -----
    Files are written to the current working directory.
    Existing files with the same name will be overwritten.
    """

    for key, arr in data_dict.items():
        filename = f"{prefix}_{key}.npy"
        np.save(filename, arr)


# -----------------------------------------------------------------------------
# NEW FUNCTION: Save only selected arrays to a specified folder
# -----------------------------------------------------------------------------
def SaveSelectedArrays(data_dict, folder_name, array_names=None, prefix=""):
    """
    Save only specified NumPy arrays from a dictionary to disk as .npy files
    in a specified folder.

    This is useful when you have a dictionary with many arrays but only
    want to save a subset of them to a specific directory.

    Parameters
    ----------
    data_dict : dict[str, np.ndarray]
        Dictionary returned by PVDataToNumpy() or PVDataToNumpySelected().

    folder_name : str
        Name of the folder where files will be saved. The folder will be
        created if it doesn't exist.

    array_names : list or set, optional
        Names of the arrays to save. If None, saves all arrays in the dictionary.

    prefix : str, optional, default="pvdata_selected"
        Prefix for output filenames.

        Files will be saved as:
            {folder_name}/{prefix}_{array_name}.npy

    Examples
    --------
    >>> # Save only force and points to a "forces" folder
    >>> SaveSelectedArrays(pv_data,
    ...                    folder_name="forces",
    ...                    array_names=['p_force', 'Points'],
    ...                    prefix="wing")

    This produces:
        forces/wing_p_force.npy
        forces/wing_Points.npy

    Notes
    -----
    The specified folder will be created if it doesn't exist.
    Existing files with the same name will be overwritten.
    """
    import os

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")

    if array_names is None:
        # Save all arrays
        arrays_to_save = data_dict.items()
    else:
        # Save only specified arrays that exist in the dictionary
        array_set = set(array_names)
        arrays_to_save = [(name, data_dict[name]) for name in array_set
                         if name in data_dict]

        # Warn about missing arrays
        missing = array_set - set(data_dict.keys())
        if missing:
            print(f"Warning: The following arrays were not found in the dictionary: {missing}")

    saved_count = 0
    for key, arr in arrays_to_save:
        # Construct full file path
        filename = os.path.join(folder_name, f"{prefix}{key}.npy")
        np.save(filename, arr)
        print(f"Saved: {filename} (shape: {arr.shape})")
        saved_count += 1

    print(f"\nSuccessfully saved {saved_count} arrays to folder: {folder_name}")

