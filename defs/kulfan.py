import numpy as np 
from scipy.special import comb

def to_cst_coeff_section(section:(np.array), N:(int)= 8, N1:(float)= 0.5, N2:(float)= 1.0):

    """
    Transforms the raw coordinates of a section (lower/upper) of an airfoil into the Bernstein polynomial coefficients, as in the Kulfan Form. 
    The passed airfoil must be of only one side (upper/lower) and must have either the blunt trailing/leading edges or sharp trailing/leading edges for both upper and lower parts.

    Args:
        section (np.array): An (m, 2) array of the (x, y) coordinates of the section
        N (int): The order of Bernstein polynomials, def = 8
        N1 (float): Leading edge exponent in the Class Function, def = 0.5
        N2 (float): Trailing edge exponent in the Class Function, def = 1.0

    Returns:
        Bernstein_coeff_section (np.array): A (N+1,) array of the fitted Bernstein coeffs
    """

    # 1- Handle the input data
    x_section = section[:,0]
    y_section = section[:,1]

    # Normalize the x-coords to have it [0,1]
    x_section = (x_section - np.min(x_section)) / (np.max(x_section) - np.min(x_section))

    # Y value at trailing edge
    te_x = np.argmax(x_section)
    te_y = y_section[te_x]

    fit_indices = np.where((x_section > 1e-6) & (x_section < 0.99999))[0] # Get the indices at which the target vector is defined
    x_fit = x_section[fit_indices]
    y_fit = y_section[fit_indices] # Only pick the x and y vals of airfoil at which the target vector is defined

    if len(x_fit==0):
        raise ValueError("No valid points on x_fit, only consists of trail and lead edges.")


    # 2- Build class function
    C_x = (x_fit**N1) * ((1-x_fit)**N2) # Class function for all the x points.
    
    
    # 3- Build target vector
    Y= (y_fit - x_fit * te_y) / C_x # The target vector. Y = (y(x) - x * y_TE) / C(x)


    # 4- Build Bernstein polynomial coefficients, with design matrix A (A_ji = K_i(x_j))
    num_points = len(x_fit)
    num_coeffs = N+1
    A = np.zeros((num_points, num_coeffs))

    for i in range(num_coeffs):
        # Binomial coefficient: comb(N,i)
        # K_i(x) = comb(N,i) * x^i * (1-x)^(N-i)
        K_i = comb(N,i) * (x_fit**i) * ((1-x_fit)**(N-i))
        A[:,i] = K_i
    
    # 5- Calculate the Bernstein coefficients given the design matrix and the target vector
    # lstsq is the exact function to be used for this, check documentation

    Bernstein_coeff_section, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)

    return Bernstein_coeff_section

def to_cst_coeff_airfoil():

    """
    Transforms the raw coordinates of a full airfoil into the Bernstein polynomial coefficients, as in the Kulfan Form.

    Args:
        airfoil (np.array): An (m, 2) array of the (x, y) coordinates of the entire airfoil
        N (int): The order of Bernstein polynomials, def = 8
        N1 (float): Leading edge exponent in the Class Function, def = 0.5
        N2 (float): Trailing edge exponent in the Class Function, def = 1.0
    
    Returns:
        Bernstein_coeffs_airfoil (np.array): A (2(N+1),) array of the fitted Bernstein coeffs for the entire airfoil
        Bernstein_coeff_upper (np.array): A (N+1,) array of the fitted Bernstein coeffs for the upper section
        Bernstein_coeff_lower (np.array): A (N+1,) array of the fitted Bernstein coeffs for the lower section
    """

    pass