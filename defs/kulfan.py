import numpy as np 
from scipy.special import comb

def to_cst_coeff_section(section:(np.array), N:(int)= 8, N1:(float)= 0.5, N2:(float)= 1.0):

    """
    Transforms the raw coordinates of a section (lower/upper) of an airfoil into the Bernstein polynomial coefficients, as in the Kulfan Form. 
    The passed airfoil must be of only one side (upper/lower) and must have either the blunt trailing/leading edges or sharp trailing/leading edges for both upper and lower parts.

    Args:
        section (np.array): An (2, m) array of the (x, y) coordinates of the section
        N (int): The order of Bernstein polynomials, def = 8, coeffs will be +1
        N1 (float): Leading edge exponent in the Class Function, def = 0.5
        N2 (float): Trailing edge exponent in the Class Function, def = 1.0

    Returns:
        Bernstein_coeff_section (np.array): A (N+1,) array of the fitted Bernstein coeffs
    """

    # 1- Handle the input data
    x_section = section[0,:]
    y_section = section[1,:]

    # Normalize the x-coords to have it [0,1]
    x_section = (x_section - np.min(x_section)) / (np.max(x_section) - np.min(x_section))

    # Y value at trailing edge
    te_x = np.argmax(x_section)
    te_y = y_section[te_x]

    fit_indices = np.where((x_section > 1e-6) & (x_section < 0.99999))[0] # Get the indices at which the target vector is defined
    x_fit = x_section[fit_indices]
    y_fit = y_section[fit_indices] # Only pick the x and y vals of airfoil at which the target vector is defined

    if x_fit.shape ==0:
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

    return Bernstein_coeff_section, te_y

def to_cst_coeff_airfoil(airfoil:(np.array), N:(int)= 8, N1:(float)= 0.5, N2:(float)= 1.0):


    """
    Transforms the raw coordinates of a full airfoil into the Bernstein polynomial coefficients, as in the Kulfan Form.

    Args:
        airfoil (np.array): An (2, m) array of the coordinates of the entire airfoil [0,:] is x coords, [1,:] is y coords
        N (int): The order of Bernstein polynomials, def = 8, coeffs will be +1
        N1 (float): Leading edge exponent in the Class Function, def = 0.5
        N2 (float): Trailing edge exponent in the Class Function, def = 1.0
    
    Returns:
        Kulfan_total (np.array): A (2(N+1)+2,) array of the fitted Bernstein coeffs (upper first) and N1 and N2 for the entire airfoil
        Bernstein_airfoil (np.array): A (2(N+1),) array of the fitted Bernstein coeffs for the entire airfoil
        Bernstein_upper (np.array): A (N+1,) array of the fitted Bernstein coeffs for the upper section
        Bernstein_lower (np.array): A (N+1,) array of the fitted Bernstein coeffs for the lower 

    """

    mid_point = int(round((airfoil.shape)[1]/2)) # Get the midpoint index of the airfoil, this is to separate upper and lower parts

    airfoil_lower = np.flip(airfoil[:,:mid_point], axis=1) # This gets the lower part of the airfoil, needs to be flipped since the dataset starts with trailing edge, 
                                                   # then goes to leading edge through the lower part, so, upper's order is correct but lower must be flipped
    airfoil_upper = airfoil[:,mid_point:] # This gets the upper part of the airfoil

    Bernstein_lower, te_y_lower = to_cst_coeff_section(section=airfoil_lower, N=N, N1=N1, N2=N2)
    Bernstein_upper, te_y_upper = to_cst_coeff_section(section=airfoil_upper, N=N, N1=N1, N2=N2)

    Kulfan_total = np.hstack((Bernstein_upper, Bernstein_lower, np.array([N1, N2, te_y_upper, te_y_lower])))
    Bernstein_airfoil = np.hstack((Bernstein_upper, Bernstein_lower))

    return Kulfan_total, Bernstein_airfoil, Bernstein_upper, Bernstein_lower

def to_coords_section(B:(np.array), N:(int), N1:(float), N2:(float), y_te:(float), num_points:(int)):

    """
    Transforms the Bernstein coefficients into regular coords given

    Args:
        B (np.array): An (N+1,) array of Bernstein coefficients
        N (int): The order of max Bernstein polynomial that was used
        N1 (float): Leading edge exponent in the Class Function
        N2 (float): Trailing edge exponent in the Class Function
        y_te (float): The y-coordinate of the section's trailing edge
        num_points (int): The number of that will be used to reconstruct
    """

    x = np.linspace(0,1,num_points)

    # Class function
    C_x = (x**N1) * ((1-x)**N2)

    # Shape function
    S_x = np.zeros_like(x)

    # This gets us the S(x)
    num_coeffs = N+1
    for i in range(num_coeffs):
        K_i = comb(N, i) * (x**i) * ((1-x)**(N-i))
        S_x += B[i] * K_i # We add the sum terms to the entire x vector each time

    # Get y(x)
    y_x = C_x * S_x + x * y_te

    # Compile them into a single coord matrix
    section = np.vstack((x, y_x))

    return section

def to_coords_airfoil(Kulfan_total:(np.array), N:(int), num_points:(int)):

    """
    Reconstructs the entire airfoil and returns an array of the airfoil where [0] is the x-coords and [1] is y-coords. 
    The coords start from TE of lower, goes to LE, then ends at TE of upper, this is how our airfoil datasets are.

    Args:
        Kulfan_total (np.array): 'Kulfan_total' output of to_cst_coeff_airfoil
        N (int): The order of polynomials used while calculating the cst form
        num_points (int): Total number of points per section that will be used to reconstruct

    Returns:
        airfoil (np.array): The np.array of the entire airfoil
        upper_section (np.array): The np.array of the upper section of the airfoil
        lower_section (np.array): The np.array of the lower section of the airfoil
    """

    # Unpack all the values, assuming to kullfan was done by to_cst_coeff_airfoil function
    num_coeffs = N+1
    B_upper = Kulfan_total[:num_coeffs]
    B_lower = Kulfan_total[num_coeffs:2*num_coeffs]
    N1 = Kulfan_total[2*num_coeffs]
    N2 = Kulfan_total[2*num_coeffs + 1]
    te_y_upper = Kulfan_total[2*num_coeffs + 2]
    te_y_lower = Kulfan_total[2*num_coeffs + 3]

    upper_section = to_coords_section(B=B_upper, N=N, N1=N1, N2=N2, y_te=te_y_upper, num_points=num_points)
    lower_section = to_coords_section(B=B_lower, N=N, N1=N1, N2=N2, y_te=te_y_lower, num_points=num_points)

    # We must flip the section to match the assumed dataset
    lower_section = np.flip(lower_section, axis=1)

    # Combine them to get the airfoil matrix back, starts with TE of lower, goes to LE, ends at TE of upper
    airfoil = np.hstack((lower_section, upper_section))

    return airfoil, upper_section, lower_section
