from __future__ import division

import sys

from numpy import arange, array, ones, identity, dot, zeros, sin, cos, pi,\
                  sqrt, sum, arccos, transpose, newaxis, tensordot
from numpy.linalg import norm
from scipy.spatial.distance import cdist

sys.path.insert(0, 'old/scalingproject')
sys.path.insert(0, 'old/src')
from topographicmap import read_electrode_locations


def calculate_lead_field(gen_conf):
    """Should be used to calculate lead field when this is done only once,
    however for better performance when the lead field is to be calculated
    multiple times, the Lead_Field class should be used, where
    initialize_electrode_locations() is calculated only once.
    """
    radius, xyz_el = initialize_electrode_locations()
    return calculate_lead_field_given_electrodes(gen_conf, radius, xyz_el)

def initialize_electrode_locations():
    """Reads in electrode locations and transforms them to xyz coordinates, so
    that this isn't unnecessarily repeated in the main lead_field calculation
    function if not strictly necessary.
    """
    # Setting the radius of the head to 11.5 cm
    radius = 11.5

    # Reading in electrode locations from external file electrodeLocations.elp
    [el, el_x, el_y, el_thetas, el_phis] = read_electrode_locations()
    
    # How many electrodes do we have?
    n_el = len(el)
    
    # Coordinates of the electrodes (in the frame of reference associated with
    # the center of the head)
    xyz_el = zeros((n_el,3))
    for i_el in range(n_el):
        # Calculating the coordinates of the electrode in the Cartesian coordinates associated with the head
        # The X axis points towards the right ear, while the Y axis points towards the front
        el_theta = el_thetas[i_el]
        el_phi = el_phis[i_el]
        xyz_el[i_el,0] = radius * sin(el_theta) * cos(el_phi);
        xyz_el[i_el,1] = radius * sin(el_theta) * sin(el_phi);
        xyz_el[i_el,2] = radius * cos(el_theta);

    return radius, xyz_el

def calculate_lead_field_given_electrodes(gen_conf, radius, xyz_el):
    """Actual calculation of lead field."""
    # Assuming ideal conductivity
    sigma = 1.0
    
    # How many generators and electrodes do we have?
    n_gen = len(gen_conf)
    n_el = xyz_el.shape[0]

    # The number of electrodes and generators defines the size of the lead field matrix
    lead_field_brody_1973 = zeros((n_el,n_gen))

    # Coordinates of each dipole in the frame of reference associated with the
    # head and the orienation of the dipole in the frame of reference
    # associated with the dipole, with axes parallel to the head frame of
    # reference.
    xyz_dipole = zeros((n_gen,3))
    xyz_orientation = zeros((n_gen,3))
    for i_gen in range(n_gen):
        # Calculating the coordinates of the dipole in the Cartesian coordinates associated with the head
        dipole_radius = radius - gen_conf[i_gen]['depth']
        dipole_theta = gen_conf[i_gen]['theta']
        dipole_phi = gen_conf[i_gen]['phi']
        cos_dipole_theta = cos(dipole_theta)
        sin_dipole_theta = sin(dipole_theta)
        cos_dipole_phi = cos(dipole_phi)
        sin_dipole_phi = sin(dipole_phi)
        xyz_dipole[i_gen,0] = dipole_radius * sin_dipole_theta * cos_dipole_phi
        xyz_dipole[i_gen,1] = dipole_radius * sin_dipole_theta * sin_dipole_phi
        xyz_dipole[i_gen,2] = dipole_radius * cos_dipole_theta
            
        # The Orientation vector
        orientation_theta = gen_conf[i_gen]['orientation']
        orientation_phi = gen_conf[i_gen]['orientation_phi']
        xyz_orientation_rotated = zeros(3);
        xyz_orientation_rotated[0] = sin(orientation_theta) * cos(orientation_phi);
        xyz_orientation_rotated[1] = sin(orientation_theta) * sin(orientation_phi);
        xyz_orientation_rotated[2] = cos(orientation_theta);
        
        # Rotation matrix for translating the coordinates in the dipole frame of reference to the
        # coordinates associated with the dipole parallel to the head coordinates.
        rotation_matrix = zeros((3,3));
        dipole_theta = gen_conf[i_gen]['theta']
        dipole_phi = gen_conf[i_gen]['phi']
        # Row 1
        rotation_matrix[0,0] = sin_dipole_phi
        rotation_matrix[0,1] = cos_dipole_theta * cos_dipole_phi
        rotation_matrix[0,2] = sin_dipole_theta * cos_dipole_phi
        # Row 2
        rotation_matrix[1,0] = -cos_dipole_phi
        rotation_matrix[1,1] = cos_dipole_theta * sin_dipole_phi
        rotation_matrix[1,2] = sin_dipole_theta * sin_dipole_phi
        # Row 3
        rotation_matrix[2,0] = 0
        rotation_matrix[2,1] = -sin_dipole_theta
        rotation_matrix[2,2] = cos_dipole_theta
            
        # Rotating orientation to translated dipole coordinates
        xyz_orientation[i_gen,:] = dot(rotation_matrix,xyz_orientation_rotated)

    distance = cdist(xyz_el, xyz_dipole)
    r_cos_phi = dot(xyz_el, transpose(xyz_dipole)) / radius

    # Coordinate arrays broadcast to shape of (n_el,n_gen,3) in order to
    # vectorize all further calculations
    xyz_el_b = xyz_el[:,newaxis,:] + zeros((n_el,n_gen,3))
    xyz_dipole_b = xyz_dipole[newaxis,:,:] + zeros((n_el,n_gen,3))

    field_vector = xyz_el_b - xyz_dipole_b
    field_vector = 2*field_vector / (distance**2)[:,:,newaxis]
    field_vector += (1/(radius**2)) * \
            ((xyz_el_b * r_cos_phi[:,:,newaxis] - radius * xyz_dipole_b) /\
             (distance - r_cos_phi + radius)[:,:,newaxis] + xyz_el_b)
    field_vector /=  4 * pi * sigma * distance[:,:,newaxis]
    
    lead_field_brody_1973 = sum(field_vector*xyz_orientation[newaxis,:,:],2)
    
    return lead_field_brody_1973


class Lead_Field:
    """The Lead_Field class should be used when calculating the lead field
    multiple times, it's performance is better than calculate_lead_field()
    because it reads in the electrode locations only once, on initialization.
    """
    def __init__(self):
        self.radius, self.xyz_el = initialize_electrode_locations()
        
    def calculate(self, gen_conf):
        return calculate_lead_field_given_electrodes(gen_conf, self.radius,\
                                                     self.xyz_el)