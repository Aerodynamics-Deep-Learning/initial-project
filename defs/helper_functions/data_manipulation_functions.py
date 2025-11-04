"""
In this file, we define the data manipulator functions that can transform the various types of data.
"""

import pandas as pd
import torch
import ast

def datareform_tuple_coordinates(df: pd.DataFrame, which_coord: str = "y"):
    """
    Conversion function that reforms the tuples and returns coordinates, sys coeffs, airfoil coeffs
    
    Parameters:
        df (pd.DataFrame): dataframe to be converted
        which_coord (str): "all"= return tuples, "y"= return y values, "x"= return x values

    Returns:
        df (pd.DataFrame): converted dataframe
    """

    def _tuplestr_to_coord_ynumber(x):
        """
        Conversion function from a str representing a tuple to y values of the tuple to a number. If there's no str, then straight up returns the val

        Parameters:
            x (any): entry in the df
        """

        if isinstance(x, str) and x.startswith("(") and x.endswith(")"):
            try:
                return (ast.literal_eval(x))[1] #literally evaluate and return the y values at dim
            except:
                raise Exception("Couldn't convert the str into a tuple and retreive the number")
            
        return x
    
    def _tuplestr_to_coord_xnumber(x):
        """
        Conversion function from a str representing a tuple to x values of the tuple to a number. If there's no str, then straight up returns the val

        Parameters:
            x (any): entry in the df
        """

        if isinstance(x, str) and x.startswith("(") and x.endswith(")"):
            try:
                return (ast.literal_eval(x))[0] #literally evaluate and return the y values at dim
            except:
                raise Exception("Couldn't convert the str into a tuple and retreive the number")
            
        return x

    def _tuplestr_to_tuple(x):
        """
        Conversion function from a str representing to a tuple. If there's no str, then straight up returns the val

        Parameters:
            x (any): entry in the df
        """

        if isinstance(x, str) and x.startswith("(") and x.endswith(")"):
            try:
                return ast.literal_eval(x) #literally evaluate and return the y values at dim
            except:
                raise Exception("Couldn't convert the str into a tuple")
            
        return x
    
    columns_to_change = df.columns[:] # Pick the columns that represent coordinates
    
    df = df[columns_to_change].astype(object) #

    if which_coord == "y":
        return df[columns_to_change].map(_tuplestr_to_coord_ynumber)
    
    elif which_coord == "x":
        return df[columns_to_change].map(_tuplestr_to_coord_xnumber)
    
    elif which_coord == "all":
        return df[columns_to_change].map(_tuplestr_to_tuple)

    else:
        raise Exception("Entered unsupported which_coord entry")


def seperate_transform(df:(pd.DataFrame), target_coord:(int)):
    """
    Makes the necessary changes to the dataset to get target tensor and input tensor.

    Parameters:
        df (pd.DataFrame): The dataframe to be parsed.
        target_coord (int): Coord at which target values start.

    Returns
        tensor_inputs (torch.Tensor): The tensor that will be fed into the nn.
        tensor_targets (torch.Tensor): The tensor that has targets.
    """

    df = df.to_numpy()

    return torch.from_numpy(df[:,:target_coord]).float(), torch.from_numpy(df[:,target_coord:]).float() # Double precision since problem requires high pres as we comparing real values


def kulfan_dataframe(df:(pd.DataFrame), airfoil_pts:(int), N:(int), N1:(float)=0.5, N2:(float)=1):

    """
    Takes in an entire dataframe, and then replaces the airfoil coordiniates with the Kulfan form coefficients.
    To make things faster, it creates a dictionary of airfoil coordinates, and if the coeffs for that airfoil 
    has been calculated before, it just picks the coeffs from the dictionary. Returns the modified dataframe.

    Args:
        df (pd.DataFrame): The dataframe of the airfoil dataset
        airfoil_pts (int): How many airfoil points are in the dataset
        N (int): The order of Bernstein polynomials that'll be used to make the transformation
        N1 (float): Leading edge exponent in the Class Function, def = 0.5 for subsonic airfoils
        N2 (float): Trailing edge exponent in the Class Function, def = 1.0 for subsonic airfoils

    Returns:
        df_kulfan (pd.DataFrame): The modified dataframe which has 
            - First row as the arbitrary name of the airfoil
            - Following rows as the Bernstein coefficients (upper and lower), N1, N2, and y_TE (upper and lower)
            - Following rows as the system conditions (Re, Mach, AoA, etc.)
            - Following rows as the target values (Cd, Cl, Cm, etc.)
    """

    


    pass

