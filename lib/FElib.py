import numpy as np
from scipy.linalg import eigh

# =======================================================
#                 ELEMENT MATRICES
# =======================================================
def elMatrixBar4DoF(E, A, rho, L):
    """
    2D bar/truss element matrices (axial deformation only)
    
    Parameters:
    -----------
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    rho : float
        Mass density
    L : float
        Element length
    tension : float, optional
        Pre-tension in the element for geometric stiffness
        
    Returns:
    --------
    K : ndarray
        Element stiffness matrix (4x4)
    M : ndarray
        Element mass matrix (4x4)
    """
    

    # Direction cosines can be added as parameters if needed
    # Here we assume they'll be applied by the calling function
    c = 1  # cos(theta)
    s = 0  # sin(theta)
    
    # Elastic stiffness matrix
    K = (E*A/L) * np.array([
        [c**2, c*s, -c**2, -c*s],
        [c*s, s**2, -c*s, -s**2],
        [-c**2, -c*s, c**2, c*s],
        [-c*s, -s**2, c*s, s**2]
    ])

    # Consistent mass matrix
    M = (rho*A*L/6) * np.array([
        [2, 0, 1, 0],
        [0, 2, 0, 1],
        [1, 0, 2, 0],
        [0, 1, 0, 2]
    ])

    return K, M

def elMatrixBeam4DoF(E, A, I, rho, L):
    """
    2D Euler-Bernoulli beam element matrices (transverse and rotational DOFs)
    
    Parameters:
    -----------
    E : float
        Young's modulus
    I : float
        Area moment of inertia
    A : float
        Cross-sectional area
    rho : float
        Mass density
    L : float
        Element length
    tension : float, optional
        Axial tension for geometric stiffness
        
    Returns:
    --------
    K : ndarray
        Element stiffness matrix (4x4)
    M: ndarray
        Element mass matrix (4x4)
    """
    # Elastic stiffness matrix
    K = (E*I/L**3) * np.array([
        [12,     6*L,    -12,     6*L],
        [6*L,   4*L**2,  -6*L,   2*L**2],
        [-12,   -6*L,     12,    -6*L],
        [6*L,   2*L**2,  -6*L,   4*L**2]
    ])

    # Consistent mass matrix
    M = (rho*A*L/420) * np.array([
        [156,    22*L,     54,    -13*L],
        [22*L,  4*L**2,   13*L,  -3*L**2],
        [54,     13*L,    156,    -22*L],
        [-13*L, -3*L**2, -22*L,   4*L**2]
    ])

    return K, M

def elMatrixBeam6DoF(E, A, I, rho, L):
    """
    2D Euler-Bernoulli beam element matrices 
    (axial, transverse and rotational DOFs)
    # --------------------------------------------
    # Elastic stiffness matrix
    # DOF order: [u1, v1, θz1, u2, v2, θz2]
    # where u = axial, v = transverse, θz = rotation about z-axis
    # --------------------------------------------
    Parameters:
    -----------
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    I : float
        Area moment of inertia
    rho : float
        Mass density
    L : float
        Element length
    tension : float, optional
        Axial tension for geometric stiffness

    Returns:
    --------
    K : ndarray
        Element stiffness matrix (6x6)
    M : ndarray
        Element mass matrix (6x6)
    """

    K = np.zeros((6, 6))
    
    # Axial stiffness component (u1, u2)
    K[0, 0] = K[3, 3] = E*A/L
    K[0, 3] = K[3, 0] = -E*A/L
    
    # Bending stiffness component (v1, θ1, v2, θ2)
    K[1, 1] = K[4, 4] = 12*E*I/L**3
    K[1, 2] = K[2, 1] = 6*E*I/L**2
    K[1, 4] = K[4, 1] = -12*E*I/L**3
    K[1, 5] = K[5, 1] = 6*E*I/L**2
    K[2, 2] = K[5, 5] = 4*E*I/L
    K[2, 4] = K[4, 2] = -6*E*I/L**2
    K[2, 5] = K[5, 2] = 2*E*I/L
    K[4, 5] = K[5, 4] = -6*E*I/L**2

    # Consistent mass matrix
    M = np.zeros((6, 6))
    
    # Axial mass component
    M[0, 0] = M[3, 3] = 2*rho*A*L/6
    M[0, 3] = M[3, 0] = rho*A*L/6
    
    # Bending mass component
    M[1, 1] = M[4, 4] = 156*rho*A*L/420
    M[1, 2] = M[2, 1] = 22*rho*A*L**2/420
    M[1, 4] = M[4, 1] = 54*rho*A*L/420
    M[1, 5] = M[5, 1] = -13*rho*A*L**2/420
    M[2, 2] = M[5, 5] = 4*rho*A*L**3/420
    M[2, 4] = M[4, 2] = 13*rho*A*L**2/420
    M[2, 5] = M[5, 2] = -3*rho*A*L**3/420
    M[4, 5] = M[5, 4] = -22*rho*A*L**2/420
    
    return K, M

def elMatrixBar6DoF(E, A, rho, L):
    """
    2D bar element matrices (axial deformation only)
    # --------------------------------------------
    # Elastic stiffness matrix
    # DOF order: [u1, v1, w1, u2, v2, w2]
    # where u = axial, v = transverse, w = vertical
    # --------------------------------------------
    Parameters:
    -----------
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    rho : float
        Mass density
    L : float
        Element length
    tension : float, optional
        Axial tension for geometric stiffness

    Returns:
    --------
    K : ndarray
        Element stiffness matrix (6x6)
    M : ndarray
        Element mass matrix (6x6)
    """ 

    K = np.zeros((6, 6))

    # Axial stiffness component (u1, u2, v1, v2, w1, w2)
    K[0, 0] = K[3, 3] = E*A/L
    K[0, 3] = K[3, 0] = -E*A/L
    K[1, 1] = K[4, 4] = E*A/L
    K[1, 4] = K[4, 1] = -E*A/L
    K[2, 2] = K[5, 5] = E*A/L
    K[2, 5] = K[5, 2] = -E*A/L
    
    # Consistent mass matrix
    M = np.zeros((6, 6))

    # Axial mass component
    M[0, 0] = M[3, 3] = 2*rho*A*L/6
    M[0, 3] = M[3, 0] = rho*A*L/6
    M[1, 1] = M[4, 4] = 2*rho*A*L/6
    M[1, 4] = M[4, 1] = rho*A*L/6
    M[2, 2] = M[5, 5] = 2*rho*A*L/6
    M[2, 5] = M[5, 2] = rho*A*L/6
    
    return K, M

def elMatrixBeam10DoF(E, A, I, rho, L):
    """
    2D Euler-Bernoulli beam element matrices 
    (axial, transverse and rotational DOFs)
    # --------------------------------------------
    # Elastic stiffness matrix
    # DOF order: [u1, v1, w1, θy1, θz1, u2, v2, w2, θy2, θz2]
    # where u = axial, v = transverse, w = vertical, 
    #       θy = rotation about y-axis, 
    #       θz = rotation about z-axis
    # --------------------------------------------
    Parameters:
    -----------
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    I : float
        Area moment of inertia
    rho : float
        Mass density
    L : float
        Element length
    nu : float
        Poisson's ratio

    Returns:
    --------
    K : ndarray
        Element stiffness matrix (10x10)
    M : ndarray
        Element mass matrix (10x10)
    """
    # ----------------------------------------------------------------------------
    K = np.zeros((10, 10))  
    
    # Axial stiffness component (u1, u2)
    K[0, 0] = K[5, 5] = E*A/L
    K[0, 5] = K[5, 0] = -E*A/L
    
    # Bending stiffness component (v1, w1, θy1, v2, w2, θy2)
    K[1, 1] = K[6, 6] = 12*E*I/L**3
    K[1, 4] = K[4, 1] = 6*E*I/L**2
    K[1, 6] = K[6, 1] = -12*E*I/L**3
    K[1, 9] = K[9, 1] = 6*E*I/L**2
    K[4, 4] = K[9, 9] = 4*E*I/L
    K[4, 6] = K[6, 4] = -6*E*I/L**2
    K[4, 9] = K[9, 4] = 2*E*I/L
    K[6, 9] = K[9, 6] = -6*E*I/L**2
    
    # Bending stiffness component in Z direction (w1, θy1, w2, θy2)
    K[2, 2] = K[7, 7] = 12*E*I/L**3
    K[2, 3] = K[3, 2] = -6*E*I/L**2  # Note: negative due to coordinate system
    K[2, 7] = K[7, 2] = -12*E*I/L**3
    K[2, 8] = K[8, 2] = -6*E*I/L**2  # Note: negative due to coordinate system
    K[3, 3] = K[8, 8] = 4*E*I/L
    K[3, 7] = K[7, 3] = 6*E*I/L**2
    K[3, 8] = K[8, 3] = 2*E*I/L
    K[7, 8] = K[8, 7] = 6*E*I/L**2

    # ----------------------------------------------------------------------------
    # Consistent mass matrix
    M = np.zeros((10, 10))

    # Axial mass component
    M[0, 0] = M[5, 5] = 2*rho*A*L/6
    M[0, 5] = M[5, 0] = rho*A*L/6
    
     # Consistent mass matrix
    M = np.zeros((10, 10))
    
    # Axial mass component (0, 5)
    M[0, 0] = M[5, 5] = 2*rho*A*L/6
    M[0, 5] = M[5, 0] = rho*A*L/6
    
    # Bending mass components
    # For v, θz DOFs (1, 4, 6, 9)
    M[1, 1] = M[6, 6] = 156*rho*A*L/420
    M[1, 4] = M[4, 1] = 22*rho*A*L**2/420
    M[1, 6] = M[6, 1] = 54*rho*A*L/420
    M[1, 9] = M[9, 1] = -13*rho*A*L**2/420
    M[4, 4] = M[9, 9] = 4*rho*A*L**3/420
    M[4, 6] = M[6, 4] = 13*rho*A*L**2/420
    M[4, 9] = M[9, 4] = -3*rho*A*L**3/420
    M[6, 9] = M[9, 6] = -22*rho*A*L**2/420
    
    # For w, θy DOFs (2, 3, 7, 8)
    M[2, 2] = M[7, 7] = 156*rho*A*L/420
    M[2, 3] = M[3, 2] = -22*rho*A*L**2/420  # Note: negative due to coordinate system
    M[2, 7] = M[7, 2] = 54*rho*A*L/420
    M[2, 8] = M[8, 2] = -13*rho*A*L**2/420  # Note: negative due to coordinate system
    M[3, 3] = M[8, 8] = 4*rho*A*L**3/420
    M[3, 7] = M[7, 3] = 13*rho*A*L**2/420
    M[3, 8] = M[8, 3] = -3*rho*A*L**3/420
    M[7, 8] = M[8, 7] = -22*rho*A*L**2/420

    # ----------------------------------------------------------------------------
    return K, M

def elMatrixBeam12DoF(E, A, I, rho, L, nu):
    """
    2D Euler-Bernoulli beam element matrices 
    (axial, transverse and rotational DOFs)
    # --------------------------------------------
    # Elastic stiffness matrix
    # DOF order: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
    # where u = axial, 
    #       v = transverse, 
    #       w = vertical, 
    #       θx = rotation about x-axis, 
    #       θy = rotation about y-axis, 
    #       θz = rotation about z-axis
    # --------------------------------------------
    Parameters:
    -----------
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    I : float
        Area moment of inertia
    rho : float
        Mass density
    L : float
        Element length
    nu : float
        Poisson's ratio

    Returns:
    --------
    K : ndarray
        Element stiffness matrix (12x12)
    M : ndarray
        Element mass matrix (12x12) 
    """
    # ----------------------------------------------------------
    # Elastic stiffness matrix
    K = np.zeros((12, 12))
    
    # Axial stiffness component (u1, u2)
    K[0, 0] = K[6, 6] = E*A/L
    K[0, 6] = K[6, 0] = -E*A/L
    
    # Bending stiffness component (v1, w1, θx1, v2, w2, θx2)
    K[1, 1] = K[7, 7] = 12*E*I/L**3
    K[1, 2] = K[2, 1] = 6*E*I/L**2
    K[1, 7] = K[7, 1] = -12*E*I/L**3
    K[1, 8] = K[8, 1] = 6*E*I/L**2
    K[2, 2] = K[8, 8] = 4*E*I/L 
    K[2, 7] = K[7, 2] = -6*E*I/L**2
    K[2, 8] = K[8, 2] = 2*E*I/L
    K[7, 8] = K[8, 7] = -6*E*I/L**2 
    
    # Torsional stiffness component (θy1, θz1, θy2, θz2)    
    G = E/(2*(1+nu));
    J = I*A;
    K[3, 3] = K[9, 9] = G*J/L
    K[3, 9] = K[9, 3] = -G*J/L


    # ----------------------------------------------------------
    # Consistent mass matrix
    M = np.zeros((12, 12))

    # Axial mass component
    M[0, 0] = M[6, 6] = 2*rho*A*L/6
    M[0, 6] = M[6, 0] = rho*A*L/6
    M[1, 1] = M[7, 7] = 2*rho*A*L/6
    M[1, 7] = M[7, 1] = rho*A*L/6
    M[2, 2] = M[8, 8] = 2*rho*A*L/6
    M[2, 8] = M[8, 2] = rho*A*L/6
    M[3, 3] = M[9, 9] = 2*rho*J/L
    M[3, 9] = M[9, 3] = rho*J/L 
    
    # Bending mass component
    M[4, 4] = M[10, 10] = 156*rho*A*L/420
    M[4, 5] = M[5, 4] = 22*rho*A*L**2/420
    M[4, 10] = M[10, 4] = 54*rho*A*L/420
    M[5, 5] = M[11, 11] = 4*rho*A*L**3/420
    
    # Torsional mass component
    M[4, 4] = M[10, 10] = 2*rho*J*L/6
    M[4, 10] = M[10, 4] = rho*J*L/6
    # ----------------------------------------------------------
    return K, M
    
# =========================================================
#                 ASSEMBLE ELEMENT MATRICES
# =========================================================

class TempNode:
    def __init__(self, coords):
        self.coords = coords
    def getCoordinates(self):
        return self.coords

def rotateMat(node1:list[float], node2:list[float]):

    # Get the coordinates of the nodes
    x1, y1, z1 = node1
    x2, y2, z2 = node2

    # Calculate element length and direction vector
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Unit vector along element axis (x' axis in local system)
    ex = np.array([dx, dy, dz]) / L
    
    # Choose a reference vector (usually global z-axis) for determining other local axes
    # If element is vertical (parallel to global z), use global x as reference
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 0, 1])
    
    # Local y' axis (perpendicular to element axis)
    ey = np.cross(ref, ex)
    ey = ey / np.linalg.norm(ey)
    
    # Local z' axis (perpendicular to x' and y')
    ez = np.cross(ex, ey)
    
    # Rotation matrix (each row is a local unit vector expressed in global coordinates)
    T = np.vstack((ex, ey, ez))
    
    return T


class FElineElement:
    """
    =========================================================
    Base class for all line-type elements (beams, bars, etc.)
    =========================================================
    """
    def __init__(self, node_ids, nodes, props):
        self.node_ids = node_ids # [node_i, node_j]
        self.nodes = nodes       #  dict: node_id -> [x, y, z]
        self.props = props       #  dict of properties (E, A, I, etc.)
        self.length = np.linalg.norm(self.nodes[1] - self.nodes[0])

    def get_DoFmap(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_stiffness(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_mass(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_transformation(self):
        raise NotImplementedError("Subclasses must implement this method")  
    
    




# =========================================================
#                 3D ELEMENTS
# =========================================================
class FEbar3d(FElineElement):

    def get_DoFmap(self):
        self.DoFmap = np.hstack([3*(self.node_ids[0]-1), 3*(self.node_ids[0]-1)+1, 3*(self.node_ids[0]-1)+2,
                                3*(self.node_ids[1]-1), 3*(self.node_ids[1]-1)+1, 3*(self.node_ids[1]-1)+2])
        return self.DoFmap
    
    def get_stiffness(self):
        k , _ = elMatrixBar6DoF(self.props['E'], self.props['A'], self.props['rho'], self.length)
        self.k = k
        return k
    
    def get_mass(self, lump=True):
        if lump:
            m = self.props['rho'] * self.props['A'] * self.length / 2 * np.eye(6)
            self.m = m
        else:
            _, m = elMatrixBar6DoF(self.props['E'], self.props['A'], self.props['rho'], self.length)
            self.m = m
        return m
    
    def get_transformation(self):
        node1 = TempNode(self.nodes[0])
        node2 = TempNode(self.nodes[1])
        T = rotateMat(node1, node2)
        # Expand transformation matrix for all DOFs
        T_full = np.zeros((6,6))
        T_full[0:3, 0:3] = T  # for node i
        T_full[3:6, 3:6] = T  # for node j

        self.T = T_full
        return T_full
        
class FEbeam3d(FElineElement):

    def get_DoFmap(self):
        return [6*(self.node_ids[0]-1), 6*(self.node_ids[1]-1)]
    
    def get_stiffness(self):
        k , _ = elMatrixBeam6DoF(self.props['E'], self.props['A'], self.props['I'], self.props['rho'], self.props['L'])
        return k    
    
    def get_mass(self, lump=True):
        if lump:
            raise NotImplementedError("Lumping not implemented for beam elements")
        else:
            _, m = elMatrixBeam6DoF(self.props['rho'], self.props['A'], self.props['I'], self.props['L'])
        return m
    
    def get_transformation(self):
        ni, nj = self.node_ids
        return rotateMat(self.nodes[self.node_ids[0]], self.nodes[self.node_ids[1]])
    




class FEtruss:
    def __init__(self, FEelements):
        self.FEelements = FEelements
       
    def get_stiffness(self):
        K = np.zeros((3*num_nodes, 3*num_nodes))
        for elem in self.FEelements:
            elem.get_stiffness()
            elem.get_transformation()
            k_temp4thisElem = elem.T.T @ elem.k @ elem.T
            dofmap = elem.get_DoFmap()
            K[np.ix_(dofmap, dofmap)] += k_temp4thisElem
        self.K = K
        return K

    def get_mass(self):
        M = np.zeros((3*num_nodes, 3*num_nodes))
        for elem in self.FEelements:
            elem.get_mass(lump=False)
            M[np.ix_(elem.DoFmap, elem.DoFmap)] += elem.m
        self.M = M
        return M







