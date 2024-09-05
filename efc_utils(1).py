from hcipy import *
import numpy as np
import json

from magaox.camera import XCam
from magaox.deformable_mirror import XDeformableMirror

import purepyindi2 as indi
import time
from magpyx.utils import ImageStream

def take_measurement(ldm, lcam, probes, amp, options={'sleep':0.1, 'num_im':10, 'num_skip':2}):
    images = []
    for probe in probes:
        
        for sp in [-1, 1]:
            ldm.actuators += sp * amp * probe
            ldm.send(sleep=options['sleep'])
            
            corrupted_image = lcam.grab_stack(options['num_skip'])
            im = lcam.grab_stack(options['num_im'])
            
            ldm.actuators -= sp * amp * probe
            images.append(im)
    
    ldm.send()
    return Field(images, lcam.grid)

def get_timestamp():
    return '{:d}'.format( int(time.time()) )

def find_center(images):
    
    I0 = np.sum(images, axis=-1)
    Ix = np.sum(images * images.grid.x, axis=-1) / I0
    Iy = np.sum(images * images.grid.y, axis=-1) / I0
   
    return np.array([Ix, Iy])

def shift_to_center(image, center):
    I0 = np.sum(image, axis=-1)
    Ix = np.sum(image * image.grid.x, axis=-1) / I0
    Iy = np.sum(image * image.grid.y, axis=-1) / I0
    
    current_center = np.array([Ix, Iy])
    shift = current_center - center
    
    interpolator = make_linear_interpolator_separated(image, fill_value=0)
    return interpolator(image.grid.shifted(-shift))

def reduce_pwp_images(images, differential_operator, center=None, pca_modes=None):
    processed_images = images.copy()
    
    if center is not None:
        processed_images = Field([shift_to_center(image, center) for image in processed_images], processed_images.grid)
    
    if pca_modes is not None:
        processed_images = processed_images - ( pca_modes.T.dot( pca_modes.dot(processed_images.T))).T
        
    return differential_operator.dot(processed_images)

def get_amplitude_correction(grid, fourier_mask):
    fr = (grid.as_('polar').r / (2*np.pi))[fourier_mask]
    sigma = 0.127 * 48
    sigma2 = 0.3 * 48
    correction = np.sqrt( np.exp(-(0.5 * fr/sigma2)**3.5)/np.exp(-(0.5 * fr/sigma)**2) )
    return correction

def get_amplitude_correction_kilodm(grid):
    sigma_x = 6.5
    sigma_y = 6.5 / np.sqrt(2.0)
    beta = 2.2

    x = grid.x / sigma_x
    y = grid.y / sigma_y
    r = np.hypot(x, y)
    
    scaling = (1.0 + 10.0 * np.exp(-r**beta/2)) / (11.0)
    
    return Field(np.sqrt(1/scaling), grid)

def window_field(data, center, width, height):
    indx = data.grid.closest_to(center)
    ind_y, ind_x = np.unravel_index(indx, data.grid.shape)

    sub_data = data.shaped[(ind_y - height//2):(ind_y + height//2), (ind_x - width//2):(ind_x + width//2)]
    sub_grid = make_pupil_grid([width, height], [width * data.grid.delta[0], height * data.grid.delta[1]])
    return Field(sub_data.ravel(), sub_grid)

def cutout_calib_pixels(data, width, height, x, y):
    grid = make_pupil_grid([width, height], [width//5, height//5])
    sub_data = data.shaped[(y - height//2):(y + height//2), (x - width//2):(x + width//2)]
    return Field(sub_data.ravel(), grid)