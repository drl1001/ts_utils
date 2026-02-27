'''
Overview of Interpolator:
    - Create RBF interpolator based on section II.IV in [1].
    - H matrix used to interpolate displacement of FEM nodes TO CFD mesh
    - H^t used to interpolate forces from CFD to forces in FEM. 
    - Compute matrices H and H_transposed in preprocessing step BEFORE running
        any CFD (using the initial UNDEFORMED FEM grid and initial CFD mesh)
    - Store these two matrices as they are huge, but they are used for every inner
        time step and do not need updating for the same wing.

Performance metrics of interpolator (including changing the RBF basis function):
    - for CFD forces to FEM forces: plot lift distribution along span of wing --> want (near) elliptical distribution
    - for FEM displacement to CFD displacement: plot twist distrbution along span to check for smoothness
        and plot LE and TE CFD cells z-coord distrbution along span to check smoothness

Notes:
    - CANNOT use scipy.interpolate even though this is very good, because we need to store 
        the matrices H and H^t but scipy.interpolate uses chunking (i.e. never calculates the
        matrix fully) so cannot store a full H matrix.

References:
[1] Allen, C., and Rendall, T. Unified Approach to CFD-CSD Interpolation and Mesh
Motion Using Radial Basis Functions. AIAA, 2007.
'''

############################################################
# TODO: test performance of different RBF basis functions, e.g. thin plate spline, Welland C2 etc.

import numpy as np
import scipy 


def _RBF(x1, x2, phi_name : str, norm_bias : tuple = (1.,1.,1.), c=1.0, alpha=1.0, r0=1.0):
    """
    Computes various radial basis functions (RBFs) between two position vectors.

    Parameters:
    -----------
    x1 : numpy.ndarray of shape (3, 1) or (3,)
        First position vector.
    x2 : numpy.ndarray of shape (3, 1) or (3,)
        Second position vector.
    phi_name : str
        The name of the basis function to compute. Valid options are:
        'gaussian', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric',
        'wendland_c0', 'wendland_c2', 'wendland_c4', 'wendland_c6', 'euclid_hat'
    norm_bias : tuple
        tuple of shape (kx, ky, kz) for norm-biasing. The larger the coefficient in a direction,
        the less influence points in that direction have on the RBF interpolation. E.g. for a wing
        the cross-section plane is most important (x-z plane), so make ky large to bias the spanwise
        direction, so points along span have less influence than points on the cross-section.
    c : float, optional
        Shape parameter for multiquadric, inverse multiquadric, and Euclid's Hat functions.
        Default is 1.0.
    alpha : float, optional
        Shape parameter for Gaussian function. Default is 1.0.
    r0 : float, optional
        Support radius for Wendland functions and Euclid's Hat. Default is 1.0.

    Returns:
    --------
    phi : float
        The value of the radial basis function between the two position vectors.

    Raises:
    -------
    ValueError
        If an invalid basis function name is provided.
        If input vectors have incompatible shapes.

    Notes:
    ------
    - For Wendland functions and Euclid's Hat, the function returns 0 for r >= r0.
    - The Euclidean distance is computed as r = ||x1 - x2||.
    """
    
    # Validate input shapes and compute Euclidean distance
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()
    
    if x1.shape != (3,) or x2.shape != (3,):
        raise ValueError(f"Input vectors must have 3 elements each. Got shapes {x1.shape} and {x2.shape}")
    
    # Apply norm-biasing
    kx, ky, kz = norm_bias
    # Compute Euclidean distance between the two points
    r = np.sqrt(kx*(x1[0]-x2[0])**2 + ky*(x1[1]-x2[1])**2 + kz*(x1[2]-x2[2])**2)
    # np.linalg.norm(x1 - x2)
    
    # Normalize distance for functions with compact support
    r_norm = r / r0
    
    if phi_name == 'gaussian':
        # phi(r) = exp(-alpha·r)
        return np.exp(-alpha * r)
    
    elif phi_name == 'thin_plate_spline':
        # phi(r) = r² ln(r)
        # Handle r=0 case to avoid log(0)
        if r == 0:
            return 0.0
        return r**2 * np.log(r)
    
    elif phi_name == 'multiquadric':
        # phi(r) = sqrt(c² + r²)
        return np.sqrt(c**2 + r**2)
    
    elif phi_name == 'inverse_multiquadric':
        # phi(r) = 1/sqrt(c² + r²)
        return 1.0 / np.sqrt(c**2 + r**2)
    
    elif phi_name == 'wendland_c0':
        # phi(r) = (1 - r)²
        # Implement positive part: return 0 where r_norm >= 1
        return (1 - r_norm)**2 if r_norm < 1 else 0.0
    
    elif phi_name == 'wendland_c2':
        # phi(r) = (1 - r)^4 (4r + 1)
        return (1 - r_norm)**4 * (4 * r_norm + 1) if r_norm < 1 else 0.0
    
    elif phi_name == 'wendland_c4':
        # phi(r) = (1 - r)^6 (35r² + 18r + 3)
        return (1 - r_norm)**6 * (35 * r_norm**2 + 18 * r_norm + 3) if r_norm < 1 else 0.0
    
    elif phi_name == 'wendland_c6':
        # phi(r) = (1 - r)^8 (32r³ + 25r² + 8r + 1)
        return (1 - r_norm)**8 * (32 * r_norm**3 + 25 * r_norm**2 + 8 * r_norm + 1) if r_norm < 1 else 0.0
    
    elif phi_name == 'euclid_hat':
        # phi(r) = pi ((1/12)r³ - r0²·r + (4/3)r0³)
        # This function has compact support [0, r0]
        return np.pi * ((1/12) * r**3 - r0**2 * r + (4/3) * r0**3) if r < r0 else 0.0
    
    else:
        raise ValueError(f"Unknown basis function: {phi_name}. Valid options are: " +
                         "'gaussian', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric', " +
                         "'wendland_c0', 'wendland_c2', 'wendland_c4', 'wendland_c6', 'euclid_hat'")





def _RBF_matrix(X1: np.ndarray, X2: np.ndarray, phi_name: str, norm_bias: tuple = (1.,1.,1.), c=1.0, alpha=1.0, r0=1.0) -> np.ndarray:
    """
    Vectorized RBF evaluation between all pairs of points in X1 and X2.
    Replaces the nested loop over _RBF(x1, x2) by computing the full
    pairwise distance matrix via broadcasting, then applying the RBF
    formula element-wise with numpy operations.

    Parameters:
    -----------
    X1 : np.ndarray, shape (M, 3)
    X2 : np.ndarray, shape (N, 3)

    Returns:
    --------
    np.ndarray, shape (M, N)  where result[i,j] = RBF(X1[i], X2[j])
    """
    kx, ky, kz = norm_bias

    # Broadcasting trick: expand dims so diff[i,j,:] = X1[i] - X2[j]
    # X1[:, np.newaxis, :] has shape (M, 1, 3)
    # X2[np.newaxis, :, :] has shape (1, N, 3)
    # diff therefore has shape (M, N, 3)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]

    # Compute biased Euclidean distance for every (i,j) pair simultaneously
    # r has shape (M, N)
    r = np.sqrt(kx * diff[:,:,0]**2 + ky * diff[:,:,1]**2 + kz * diff[:,:,2]**2)
    r_norm = r / r0  # normalised radius used by compact-support (Wendland) functions

    # Apply the chosen RBF formula element-wise to the full (M, N) distance matrix
    if phi_name == 'gaussian':
        return np.exp(-alpha * r)

    elif phi_name == 'thin_plate_spline':
        # r=0 gives 0*log(0) which is 0 by convention; suppress the numpy warning
        with np.errstate(divide='ignore', invalid='ignore'):
            result = r**2 * np.log(r)
        result[r == 0] = 0.0
        return result

    elif phi_name == 'multiquadric':
        return np.sqrt(c**2 + r**2)

    elif phi_name == 'inverse_multiquadric':
        return 1.0 / np.sqrt(c**2 + r**2)

    elif phi_name == 'wendland_c0':
        # np.where applies the formula only where r_norm < 1, else 0 (compact support)
        return np.where(r_norm < 1, (1 - r_norm)**2, 0.0)

    elif phi_name == 'wendland_c2':
        return np.where(r_norm < 1, (1 - r_norm)**4 * (4 * r_norm + 1), 0.0)

    elif phi_name == 'wendland_c4':
        return np.where(r_norm < 1, (1 - r_norm)**6 * (35 * r_norm**2 + 18 * r_norm + 3), 0.0)

    elif phi_name == 'wendland_c6':
        return np.where(r_norm < 1, (1 - r_norm)**8 * (32 * r_norm**3 + 25 * r_norm**2 + 8 * r_norm + 1), 0.0)

    elif phi_name == 'euclid_hat':
        return np.where(r < r0, np.pi * ((1/12) * r**3 - r0**2 * r + (4/3) * r0**3), 0.0)

    else:
        raise ValueError(f"Unknown basis function: {phi_name}. Valid options are: " +
                         "'gaussian', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric', " +
                         "'wendland_c0', 'wendland_c2', 'wendland_c4', 'wendland_c6', 'euclid_hat'")


def _create_Afs(x : np.ndarray, a : np.ndarray, phi_name : str, norm_bias : tuple = (1.,1.,1.), c=1.0, alpha=1.0, r0=1.0) -> np.ndarray:
    '''
    Assemble matrix A_fs using Eqn 12 in [1]

    Parameters:
    -----------
    x : np.ndarray; shape = (N_s, 3)
        array containing x_s, y_s, z_s: the original N_s structural FEM node coordinates.
        Each row contains one node coordinate (xs, ys, zs), N_s rows in total.
    a : np.ndarray; shape = (N_a, 3)
        array containing x_a, y_a, z_a: the original N_a SURFACE aerodynamic node coordinates of the CFD mesh.
        I.e. only using the SURFACE CFD mesh, NOT the whole volume mesh.
    phi_name : str
        RBF basis function to use. Valid options are:
        'gaussian', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric',
        'wendland_c0', 'wendland_c2', 'wendland_c4', 'wendland_c6', 'euclid_hat'
    norm_bias : tuple
        tuple of shape (kx, ky, kz) for norm-biasing. The larger the coefficient in a direction,
        the less influence points in that direction have on the RBF interpolation. E.g. for a wing
        the cross-section plane is most important (x-z plane), so make ky large to bias the spanwise
        direction, so points along span have less influence than points on the cross-section.
    c : float, optional
        Shape parameter for multiquadric, inverse multiquadric, and Euclid's Hat functions.
        Default is 1.0.
    alpha : float, optional
        Shape parameter for Gaussian function. Default is 1.0.
    r0 : float, optional
        Support radius for Wendland functions and Euclid's Hat. Default is 1.0.


    Returns:
    --------
    A_fs : np.ndarray; shape = (N_a, N_s + 4)

    '''

    x_s = x[:,0] # x coords of all the N_s FEM nodes in original/undeformed structure
    y_s = x[:,1] # y coords ...
    z_s = x[:,2] # z coords ...

    x_a = a[:,0] # x coords of the N_a aerodynamci SURFACE nodes of the CFD mesh
    y_a = a[:,1]
    z_a = a[:,2]

    x_transpose = x.T
    a_transpose = a.T # transpose of A matrix shape (3, N_a)

    # number of structural nodes, N_s
    N_s = len(x_s)
    # number of aerodynamic surface nodes, N_a
    N_a = len(x_a)


    A_fs = np.zeros((N_a, N_s+4)) # initialise empty A_fs matrix

    A_fs[:,0] = 1 # first column of A_fs is 1
    A_fs[:,1:4] = a # columns 2 to 4 are x_a, y_a, z_a

    # Columns 4 onward: phi(a_i, x_j) for all (i,j) pairs at once
    A_fs[:,4:] = _RBF_matrix(a, x, phi_name=phi_name, norm_bias=norm_bias, c=c, alpha=alpha, r0=r0)

    # for i in range(N_a): # for each row (N_a rows in total)
    #     for j in range(N_s): # columns 4 to end - compute RBFs
    #         a_i = a[i] # coords of ith surface aerodynamic node
    #         s_j = x[j] # coords of jth structural node
    #         phi = _RBF(x1=a_i,
    #                    x2=s_j,
    #                    phi_name=phi_name,
    #                    norm_bias=norm_bias,
    #                    c=c,
    #                    alpha=alpha,
    #                    r0=r0,
    #                 ) # compute RBF phi_ai_sj using Eqn [12]
    #         A_fs[i,j+4] = phi

    return A_fs

def _create_Css(x : np.ndarray, phi_name : str, norm_bias : tuple = (1.,1.,1.), c=1.0, alpha=1.0, r0=1.0) -> np.ndarray:
    '''
    Assembles matrix C_ss according to Eq 10 in [1].

    Parameters:
    -----------
    x : np.ndarray; shape = (N_s, 3)
        array containing x_s, y_s, z_s: the original N_s structural FEM node coordinates.
        Each row contains one node coordinate (xs, ys, zs), N_s rows in total.
    phi_name : str
        RBF basis function to use. Valid options are:
        'gaussian', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric',
        'wendland_c0', 'wendland_c2', 'wendland_c4', 'wendland_c6', 'euclid_hat'
    c : float, optional
        Shape parameter for multiquadric, inverse multiquadric, and Euclid's Hat functions.
        Default is 1.0.
    norm_bias : tuple
        tuple of shape (kx, ky, kz) for norm-biasing. The larger the coefficient in a direction,
        the less influence points in that direction have on the RBF interpolation. E.g. for a wing
        the cross-section plane is most important (x-z plane), so make ky large to bias the spanwise
        direction, so points along span have less influence than points on the cross-section.
    alpha : float, optional
        Shape parameter for Gaussian function. Default is 1.0.
    r0 : float, optional
        Support radius for Wendland functions and Euclid's Hat. Default is 1.0.

    Returns:
    --------
    C_ss : np.ndarray; shape = (N_s + 4, N_s + 4)
    
    '''

    x_s = x[:,0] # x coords of all the N_s FEM nodes in original/undeformed structure
    y_s = x[:,1] # y coords ...
    z_s = x[:,2] # z coords ...

    N_s = len(x_s)

    # initialise empty C_ss matrix
    C_ss = np.zeros((N_s+4, N_s+4))

    # assemble top 4 rows and first 4 columns
    C_ss[0,4:] = 1
    C_ss[4:,0] = 1   
    C_ss[1:4,4:] = x.T
    C_ss[4:, 1:4] = x

    # Bottom-right block: phi(x_i, x_j) for all (i,j) pairs at once
    C_ss[4:, 4:] = _RBF_matrix(x, x, phi_name=phi_name, norm_bias=norm_bias, c=c, alpha=alpha, r0=r0)

    # for i in range(4, N_s+4):
    #     for j in range(4, N_s+4):
    #         s_i = x[i]
    #         s_j = x[j]
    #     C_ss[i,j] = _RBF(x1=s_i, x2=s_j, phi_name=phi_name)
    # for i in range(N_s):
    #     for j in range(N_s):
    #         C_ss[i+4, j+4] = _RBF(
    #             x1=x[i],
    #             x2=x[j],
    #             phi_name=phi_name,
    #             norm_bias=norm_bias,
    #             c=c,
    #             alpha=alpha,
    #             r0=r0,
    #         )

    return C_ss
        
        

def create_H(x : np.ndarray, a : np.ndarray, phi_name : str, norm_bias : tuple = (1.,1.,1.),c=1.0, alpha=1.0, r0=1.0) -> np.ndarray:
    '''
    Assemble the matrix H using Eqns 13 - 15 in [1]: H = A_fs @ C_ss^-1
    This is to compute Eqn 16 in [1]: X_a = H @ X_s -> interpolating the N_s structural nodes 
    onto the N_a aerodynamic SURFACE nodes in the CFD mesh.

    Parameters:
    -----------
    x : np.ndarray; shape = (N_s,3)
        array containing x_s, y_s, z_s: the original N structural FEM node coordinates.
    a : np.ndarray; shape = (N_a,3)
        array containing x_a, y_a, z_a: the original aerodynamic node coordinates of the CFD mesh for N structural nodes.
    phi_name : str
        RBF basis function to use. Valid options are:
        'gaussian', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric',
        'wendland_c0', 'wendland_c2', 'wendland_c4', 'wendland_c6', 'euclid_hat'
    norm_bias : tuple
        tuple of shape (kx, ky, kz) for norm-biasing. The larger the coefficient in a direction,
        the less influence points in that direction have on the RBF interpolation. E.g. for a wing
        the cross-section plane is most important (x-z plane), so make ky large to bias the spanwise
        direction, so points along span have less influence than points on the cross-section.
    c : float, optional
        Shape parameter for multiquadric, inverse multiquadric, and Euclid's Hat functions.
        Default is 1.0.
    alpha : float, optional
        Shape parameter for Gaussian function. Default is 1.0.
    r0 : float, optional
        Support radius for Wendland functions and Euclid's Hat. Default is 1.0.

        
    Returns:
    --------
    H : np.ndarray; shape = (N_a, N_s + 4)
        Matrix H for interpolation
    '''

    # Create matrix A_fs
    A_fs = _create_Afs(
        x=x,
        a=a,
        phi_name=phi_name,
        norm_bias=norm_bias,
        c=c,
        alpha=alpha,
        r0=r0
    )
    
    # Create matrix C_ss
    C_ss = _create_Css(
        x=x,
        phi_name=phi_name,
        norm_bias=norm_bias,
        c=c,
        alpha=alpha,
        r0=r0,
    )

    # Compute inverse of C_ss
    C_ss_inv = np.linalg.inv(C_ss)

    # Compute matrix H
    H = A_fs @ C_ss_inv

    return H



def build_Ms(ms: np.ndarray) -> np.ndarray:
    '''
    Build matrix Ms = [Xs Ys Zs] where each column is the x (or y or z)
    coordinates of the 3D structural GRID nodes.

    Each coordinate column is padded with four zeros at the top:

        Xs = [0, 0, 0, 0, xs]
        Ys = [0, 0, 0, 0, ys]
        Zs = [0, 0, 0, 0, zs]

    The four padded zeros are used later to accommodate the linear
    polynomial term, which allows exact recovery of translation
    and rotation.

    Parameters
    ----------
    ms : numpy.ndarray
        Array of shape (sn, 3), where sn is the number of structural
        GRID nodes in the full 3D structure.

        ms = [xs ys zs], where:
            xs = x-coordinates (sn,)
            ys = y-coordinates (sn,)
            zs = z-coordinates (sn,)

    Returns
    -------
    Ms : numpy.ndarray
        Array of shape (4 + sn, 3) defined as:

            Ms = [Xs Ys Zs]

        where each column is padded with four zeros on top.
    '''

    # ----------------------------
    # Input validation
    # ----------------------------
    if not isinstance(ms, np.ndarray):
        raise TypeError("ms must be a numpy.ndarray")

    if ms.ndim != 2:
        raise ValueError("ms must be a 2D array of shape (sn, 3)")

    if ms.shape[1] != 3:
        raise ValueError(
            f"ms must have exactly 3 columns (x, y, z). "
            f"Got shape {ms.shape}."
        )

    sn = ms.shape[0]

    # ----------------------------
    # Construct padded matrix
    # ----------------------------
    Ms = np.zeros((sn + 4, 3), dtype=ms.dtype)

    # Fill rows 4: with original coordinates
    Ms[4:, :] = ms

    return Ms




