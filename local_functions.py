__all__ = ["tissue_densitities", "mass_attenuation_coeffs_50keV", "mass_attenuation_coeffs_150keV", "calculate_discrete_phantom", "calculate_line_integrals", "define_thorax_phantom", "display_phantom", "filtered_back_projection"]

# ---------------------------------------------------------------------------------------------------------------------
# Import libraries
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from types import SimpleNamespace
from pint import Quantity
from scipy.signal import convolve

# ---------------------------------------------------------------------------------------------------------------------
# Define tissue density and mass attenuation coefficients for 50 and 150 keV
# ---------------------------------------------------------------------------------------------------------------------

tissue_densities = dict(
    blood=Quantity(1.060, "g/cm**3"),
    bone=Quantity(1.920, "g/cm**3"),
    lung=Quantity(0.001, "g/cm**3"),
    muscle=Quantity(1.050, "g/cm**3"),
)

mass_attenuation_coeffs_50keV = dict(
    blood=Quantity(0.228, "cm**2/g"),
    bone=Quantity(0.424, "cm**2/g"),
    lung=Quantity(0.208, "cm**2/g"),
    muscle=Quantity(0.226, "cm**2/g"),
)

mass_attenuation_coeffs_150keV = dict(
    blood=Quantity(0.149, "cm**2/g"),
    bone=Quantity(0.186, "cm**2/g"),
    lung=Quantity(0.154, "cm**2/g"),
    muscle=Quantity(0.169, "cm**2/g"),
    
)

# ---------------------------------------------------------------------------------------------------------------------
# Calculate linear attenuation coefficients for 50 and 150 keV
# ---------------------------------------------------------------------------------------------------------------------

lin_attenuation_coeffs_50keV = {k: tissue_densities[k] * mass_attenuation_coeffs_50keV[k] for k in tissue_densities.keys()}
lin_attenuation_coeffs_150keV = {k: tissue_densities[k] * mass_attenuation_coeffs_150keV[k] for k in tissue_densities.keys()}

# ---------------------------------------------------------------------------------------------------------------------
# Calculate discrete phantom defined by set of ellipses and linear attenuation coefficients
# ---------------------------------------------------------------------------------------------------------------------
def calculate_discrete_phantom(matrix: Tuple[int, int], phantom: List[SimpleNamespace]) -> np.ndarray:
    
    """ Calculates discretized representation of phantom 
    :param matrix:  (x, y)
    :param phantom: list of SimpleNamespaces that each must contain values for (x0, y0, a, b, theta, mue)
                    to describe a single ellipse
    :image: discretized phantom 
    """
    
    # Coordinates in matrix
    xcoords, ycoords = np.meshgrid(np.arange(-np.fix(matrix[0] / 2), np.fix(matrix[0] / 2)),
                                   np.arange(-np.fix(matrix[1] / 2), np.fix(matrix[1] / 2)))
    
    # Stack matrix coordinates into vector
    phantom_coordinates = np.stack([xcoords, ycoords], axis=-1)
    
    # Placeholder image
    image = Quantity(np.zeros(matrix), "1/cm")
    
    # Loop over all ellipses defined in phantom
    for ellipse in phantom:
        
        # Get theta
        theta = np.deg2rad(ellipse.theta)

        # For all matrix points compute distance to ellipse center
        x_ = phantom_coordinates - np.stack([ellipse.x0, ellipse.y0]).reshape(1, 1, 2)
        
        # Fill diagonal matrix D 
        D = np.array([[1/ellipse.a, 0], [0, 1/ellipse.b]])
        
        # Fill rotation matrix Q 
        Q = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        
        # Calculate D * Q
        DQ = np.einsum("ij, jk", D, Q)
        
        # Calculate (D*Q*x_)^2 (see equation in hints for exercise XCT-1)
        value =  np.sum(np.einsum("ij, xyj -> xyi", DQ, x_) ** 2, axis=-1)
        
        # Find inside of ellipse given by (D*Q*x_)^2
        mask = np.where(value < 1, np.ones_like(value), np.zeros_like(value))

        # Fill matrix points corresponding to ellipses 
        image = image + Quantity(mask, "dimensionless") * ellipse.mue
        
    return image

# ---------------------------------------------------------------------------------------------------------------------
# Calculate line integrals of "X-ray photons" cutting through ellipses
# ---------------------------------------------------------------------------------------------------------------------
def calculate_line_integrals(detector_bins: int, phi: np.array, phantom: List[SimpleNamespace]) -> np.ndarray:
    
    """ Calculate line integrals of a collection of ellipses by finding the intersection points of line with ellipses. 
    
    :param detector_bins: number of detector bins
    :param phi: sequence of angles in degree for which the integration is performed
    :param phantom: list of SimpleNamespaces that each must contain values for (x0, y0, a, b, theta, mue)
                    to describe a single ellipsis.
    :return: projections for each angle phi (detector_bins, len(phi))
    """
    
    # Polar coordinates
    r, phi_ = np.meshgrid(np.arange(-np.fix(detector_bins/2), np.fix(detector_bins/2)), np.deg2rad(phi))
    
    # Convert from polar to Cartesian coordinates 
    sin_phi = np.sin(phi_)
    cos_phi = np.cos(phi_)
    x = r * cos_phi
    y = r * sin_phi
    
    # Placeholder projection
    projection = np.zeros_like(r)
    
    # Loop over all ellipses defined in phantom
    for ellipse in phantom:
        
        # Get theta
        theta = np.deg2rad(ellipse.theta)
        
        # Calculate D*Q
        DQ = np.array([[np.cos(theta)/ellipse.a, np.sin(theta)/ellipse.a],
                      [-np.sin(theta)/ellipse.b, np.cos(theta)/ellipse.b]])
        r0 = np.stack([x - ellipse.x0, y - ellipse.y0], axis=-1)

        # Compute reoccuring terms for quadratic equation
        DQphi = np.einsum("ij, psj -> psi", DQ, np.stack([sin_phi, -cos_phi], axis=-1))
        DQr0 = np.einsum("ij, psj -> psi", DQ, r0)
        
        # Solve quadratic equation 
        A = np.sum(DQphi ** 2, axis=-1)
        B = 2 * np.sum(DQphi * DQr0, axis=-1)
        C = np.sum(DQr0 ** 2, axis=-1) - 1
        equ = B**2 - 4 * A * C
        
        # Determine two intersection points between line and ellipse
        index = np.where(equ > 0)
        solution_plus = 0.5 * (-B[index] + np.sqrt(equ[index])) / A[index]
        solution_minus = 0.5 * (-B[index] - np.sqrt(equ[index])) / A[index]
        
        # Given the intersection coordinates calculate the value of the integral
        # Assume 1 pixel = 1 mm
        proj = ellipse.mue.m * np.abs(solution_plus - solution_minus)
           
        # Fill matrix points and normalize with detector bins
        projection[index] += proj / detector_bins
        
    return projection

# ---------------------------------------------------------------------------------------------------------------------
# Define thorax phantom
# ---------------------------------------------------------------------------------------------------------------------
def define_thorax_phantom(anode_voltage: str) -> List[SimpleNamespace]:
    
    """ Define a simple thorax phantom (see exercise XCT-1)
    
    :param anode_voltage: string specifing anode voltage. Must be one of ['50kev', '150kev']
    """
    
    if anode_voltage.lower() == "50kev":
        lin_dict = lin_attenuation_coeffs_50keV
    elif anode_voltage.lower() == "150kev":
        lin_dict = lin_attenuation_coeffs_150keV
    else:
        raise ValueError("Invalid anode voltage specified. Valid options are ['50kev', '150kev']")
        
    phantom = [
        SimpleNamespace(x0=0, y0=0, a=90, b=80, theta=0, mue=lin_dict["muscle"]),
        SimpleNamespace(x0=0, y0=0, a=70, b=60, theta=0, mue=lin_dict["lung"] - lin_dict["muscle"]),
        SimpleNamespace(x0=110, y0=0, a=15, b=15, theta=0, mue=lin_dict["muscle"]),
        SimpleNamespace(x0=110, y0=0, a=5, b=5, theta=0, mue=lin_dict["bone"] - lin_dict["muscle"]),
        SimpleNamespace(x0=-110, y0=0, a=15, b=15, theta=0, mue=lin_dict["muscle"]),
        SimpleNamespace(x0=-110, y0=0, a=5, b=5, theta=0, mue=lin_dict["bone"] - lin_dict["muscle"]),
        SimpleNamespace(x0=0, y0=0, a=10, b=10, theta=0, mue=lin_dict["blood"] - lin_dict["lung"]),
        SimpleNamespace(x0=30, y0=25, a=25, b=20, theta=35, mue=lin_dict["muscle"] - lin_dict["lung"]),
    ]
    return phantom

# ---------------------------------------------------------------------------------------------------------------------
# Display phantom
# ---------------------------------------------------------------------------------------------------------------------
def display_phantom(phantom: List[SimpleNamespace], axis: plt.Axes, figure: plt.Figure, matrix: Tuple[int, int] = (256, 256)):
    
    """ Displays discretized version of a given phantom 
    
    :param phantom: Phantom definition as returned in define_thorax_phantom
    :param axis: instance of a matplotlib axis
    :param figure: instance of the matplotlib figure containing matplotlib axis
    :param matrix: (int, int) 
    """
    
    discrete_phantom = calculate_discrete_phantom(matrix, phantom)
    
    artist = axis.imshow(discrete_phantom.m_as("1/cm"), origin="lower", cmap="gray")
    figure.colorbar(artist, ax=axis, label="Linear attenuation coefficient [1/cm]")

    xticks = np.unique(np.concatenate([np.linspace(0, matrix[0]//2, 4), np.linspace(matrix[0]//2, matrix[0], 4)]))
    yticks = np.unique(np.concatenate([np.linspace(0, matrix[1]//2, 4), np.linspace(matrix[1]//2, matrix[1], 4)]))
    xtick_labels = [f"{t:1.0f}" for t in xticks - matrix[0]//2]
    ytick_labels = [f"{t:1.0f}" for t in yticks - matrix[1]//2]
    
    axis.set_yticks(yticks), axis.set_xticks(xticks);
    axis.set_yticklabels(ytick_labels), axis.set_xticklabels(xtick_labels);
    axis.set_xlabel("x"), axis.set_ylabel("y");

# ---------------------------------------------------------------------------------------------------------------------
# Filtered backprojection
# ---------------------------------------------------------------------------------------------------------------------
def filtered_back_projection(sinogram: np.ndarray, matrix: Tuple[int, int], projection_angles: np.ndarray) -> np.ndarray:
   
    """ Performs filtered backprojection reconstruction using sinogram as input.
    
    :param sinogram: array containing the sinogram with shape (#projection_angles, #detector_bins)
    :param matrix: (int, int) size of the targeted reconstruction matrix
    :param projection_angles: angles in degree corresponding to the 0th axis of the sinogram
    """
    
    # Image matrix
    matrix = np.array(matrix)
    
    # Define highpass filter in Fourier domain
    filt = np.abs(np.arange(np.fix(-matrix[0] / 2), np.fix(matrix[0] / 2)))
    
    # Define lowpass weighting
    weight = 0.5 + np.cos(filt / matrix[0] * 2 * np.pi) /2 
    
    # Calculate highpass-lowpass filter
    filt = filt * weight**2
    
    # Define combined filter in image domain 
    filt_ = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(filt))).real / 2
    
    # Image matrix coordinates
    x, y = np.meshgrid(np.arange(-np.floor(matrix[0] / 2), np.ceil(matrix[0] / 2)),
                       np.arange(-np.floor(matrix[1] / 2), np.ceil(matrix[1] / 2)), 
                       indexing="ij")    

    # Image matrix
    result = np.zeros(matrix, dtype=np.float64)
   
    # Loop over all projection angles
    for idx, phi in enumerate(projection_angles):
        
        # Convert from x-y to r-s coordinate (see figure in XCT exercise 1)
        rs = np.around(x * np.sin(np.deg2rad(phi)) + y * np.cos(np.deg2rad(phi)))
        
        # Convert signed rs indices to unsigned rs indices 
        rs = (rs + np.ceil((sinogram.shape[1]) / 2)).astype(int)
        
        # Find rs indices for which sinogram values exist
        ix = np.where(np.logical_and(rs >= 0, rs <= sinogram.shape[1] - 1))
        
        # Copy of sinogram values for given angle phi
        sino = sinogram[idx]
        
        # Perform convolution with highpass filter
        filtered_sino = convolve(sino, filt_, 'same')
    
        # Add back-projected sinogram values
        result[ix] = result[ix] + filtered_sino[rs[ix]]  
        
    return result.T

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------