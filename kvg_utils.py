from magpyx.utils import ImageStream


def get_circular_mask(ceny, cenx, iwa, owa, pixscale, shape):
    idy, idx = np.indices(shape, dtype=float)
    idy -= ceny
    idx -= cenx
    idy = idy / pixscale # to lambda/D
    idx = idx / pixscale
    
    r_lambdaD = np.sqrt(idx**2 + idy**2)
    return (r_lambdaD >= iwa) & (r_lambdaD <= owa)

def get_semicircular_mask(ceny, cenx, iwa, owa, pixscale, shape, angle, iwa_margin=0):
    
    circ_mask = get_circular_mask(ceny, cenx, iwa, owa, pixscale, shape)
    
    idy, idx = np.indices(shape)
    
    angle_rad = np.deg2rad(angle)
    idxx = ((idx-cenx) * np.cos(angle_rad) - (idy-ceny) * np.sin(angle_rad))/pixscale
    idyy = ((idx-cenx) * np.sin(angle_rad) + (idy-ceny) * np.cos(angle_rad))/pixscale
    #theta = np.rad2deg(np.arctan2(idy - ceny - np.sin(angle_rad) * iwa_margin * iwa * pixscale,
    #                              idx - cenx - np.cos(angle_rad) * iwa_margin * iwa * pixscale)) + 180
    #theta_mask = (theta >= (angle)) & (theta < (angle + 180))
    #return theta
    mask = idxx >= (iwa_margin*iwa)
    
    return mask * circ_mask

def get_radial_dist(shape, scaleyx=(1.0, 1.0), cenyx=None):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    if cenyx is None:
        cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def get_radial_contrast(im, mask, nbins=50, cenyx=None):
    radial = get_radial_dist(im.shape, cenyx=cenyx)
    bins = np.linspace(0, radial.max(), num=nbins, endpoint=True)
    digrad = np.digitize(radial, bins)
    profile = np.asarray([np.mean(im[ (digrad == i) & mask]) for i in np.unique(digrad)])
    return bins, profile

def compute_offset(imref, image, mask, upsample=100):
    (y,x),_,_ = phase_cross_correlation(imref*mask, image*mask,
                                        upsample_factor=upsample,
                                        normalization=None)
    return y,x

def get_core_centroid(im, mask):
    com_yx = center_of_mass(im * mask)
    return int(np.rint(com_yx[0])), int(np.rint(com_yx[1]))

def get_speckle_centroid(im, mask):
    ymax, xmax = np.where( (im*mask) == (im*mask).max() )
    return np.squeeze([ymax[0], xmax[0]])

def autocalibrate_pre(im_aligned, camstream, client, calib_params, realign=True, satval=55000, skip_fsm=False, manual_intervention=False):
    '''
    Basic idea:
    * Drive PSF off-axis
    * Adjust settings (gain, exposure time, attenuation, etc) such that core is not saturated
    * Image 1: non-saturated core
    * Adjust settings with KNOWN scaling (exposure time, attenuation) to get high SNR on speckles
    * Image 2: high-SNR speckles (but saturated core)
    * Then adjust settings with UNKNOWN scaling (gain, anything else) to the mode they'll be fixed at for DH digging
    * Image 3: high-SNR speckles in DH mode
    
    Then, what we want to know is the contrast normalization for the final settings (image 3)
    * ratio of speckles in image 3 to image 2 gives attenuation factor for unknown settings = atten
    * ratio of speckles in image 2 to core in image 1 gives contrast of the speckle = contrast
    So that the normalization is
    * contrast / attenuation
    
    '''
    
    # drive PSF off-axis
    if not skip_fsm:
        print('Driving PSF off-axis')
        move_relative(client, 'stagepiezo.stagepupil_x_pos', 200)
        sleep(5)
        move_relative(client, 'stagepiezo.stagepupil_y_pos', 200)
        sleep(5)

    print('Adjusting for non-saturated core')
    camstream.set_attenuation(calib_params['atten_min'], wait=5)
    camstream.set_exposure_time(calib_params['exp_min'], wait=5)
    camstream.set_gain(calib_params['gain_min'], wait=5)
        
    print('Taking measurements at settings for non-saturated core')
    im_core = np.mean(camstream.grab_many(100), axis=0)
    
    im_core_bg = take_bg(n=100, nwait=20)

    print('Adjusting settings with known scaling for high SNR speckle measurement')
    camstream.set_attenuation(calib_params['atten_max'], wait=5)
    camstream.set_exposure_time(calib_params['exp_max'], wait=5)
    im_speckle = np.mean(camstream.grab_many(100), axis=0)
    
    im_speckle_bg = take_bg(n=100, nwait=20)
    
    print('Going to DH-digging settings (settings with unknown scaling) to get attenuation factor')
    if manual_intervention:
        input('Awaiting manual intervention (adjust to DH settings)')
    camstream.set_gain(calib_params['gain_max'], wait=5)
    camstream.set_exposure_time(calib_params['exp_dh'], wait=5) # make this adjustable!!!!!
    im_dh = np.mean(camstream.grab_many(100), axis=0)
    
    im_dh_bg = take_bg(n=100, nwait=20)
    
    return im_core, im_speckle, im_dh, im_core_bg, im_speckle_bg, im_dh_bg


def process_calibration_measurements(im_core, im_speckle, im_dh, calib_params, skip_conv=False):
    '''
    Use the measured attenuation in the reference speckle to determine the absolute counts per second in the core
    in the DH polarization state from the measured core in the calibration polarization state
    
    im_calibpol_tmax_gmax
    '''
    plate_scale = calib_params['plate_scale']
    
    # they're already in counts per sec
    core_calib_cps = im_core 
    speckle_calib_cps = im_speckle 
    speckle_dh_cps = im_dh
    
    # pick out the slices
    coreyx = get_core_centroid(im_core, calib_params['mask_core'])
    speckleyx = get_speckle_centroid(im_speckle, calib_params['mask_speckle'])
    
    extent_speckle = calib_params['cutout_extent_speckle']
    extent_core = calib_params['cutout_extent_core']
    speckle_slice = (slice(speckleyx[0]-extent_speckle//2, speckleyx[0]+extent_speckle//2),
                     slice(speckleyx[1]-extent_speckle//2, speckleyx[1]+extent_speckle//2))
    core_slice = (slice(coreyx[0]-extent_core//2, coreyx[0]+extent_core//2),
                  slice(coreyx[1]-extent_core//2, coreyx[1]+extent_core//2))
    print(speckle_slice, core_slice)
        
    # convolve with lambda/D gaussian(?) kernel
    apkernel = iefc.get_aperture_kernel(plate_scale, core_calib_cps.shape, oversample=16)
    
    if not skip_conv:
        core_calib_cps_apconv = imutils.convolve_fft(core_calib_cps, apkernel, force_real=True)
        speckle_calib_cps_apconv = imutils.convolve_fft(speckle_calib_cps, apkernel, force_real=True)
        speckle_dh_cps_apconv = imutils.convolve_fft(speckle_dh_cps, apkernel, force_real=True)
    else:
        core_calib_cps_apconv = core_calib_cps
        speckle_calib_cps_apconv = speckle_calib_cps
        speckle_dh_cps_apconv = speckle_dh_cps
    
    # compute the attenuation (due to gain, polarization, etc)
    # replace max with sum here
    atten = speckle_calib_cps_apconv[speckle_slice].sum()  / speckle_dh_cps_apconv[speckle_slice].sum()
    
    # compute the core CPS in the DH polarization
    core_cps_dh = core_calib_cps_apconv[core_slice].max() / atten
    
    # return all the things
    return {
        'core_slice' : core_slice,
        'speckle_slice' : speckle_slice,
        'core_calib_cps_apconv' : core_calib_cps_apconv,
        'speckle_calib_cps_apconv' : speckle_calib_cps_apconv,
        'speckle_dh_cps_apconv' : speckle_dh_cps_apconv,
        'atten' : atten,
        'core_cps_dh' : core_cps_dh,
        'coreyx' : coreyx,
        'speckleyx' : speckleyx
    }
    
    
def display_calibration(calib_proc, im_core, im_speckle, im_dh):
    #cross checks of everything
    coreyx = calib_proc['coreyx']
    speckleyx = calib_proc['speckleyx']
    core_slice = calib_proc['core_slice']
    speckle_slice = calib_proc['speckle_slice']
    
    
    # plot of one raw image, with core and speckle positions marked
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(calib_proc['speckle_calib_cps_apconv'], norm=LogNorm())
    coreyx = calib_proc['coreyx']
    speckleyx = calib_proc['speckleyx']
    ax.scatter(coreyx[1], coreyx[0], marker='X', c='C1', label='core')
    ax.scatter(speckleyx[1], speckleyx[0], marker='X', c='C3', label='ref speckle')
    fig.legend()
    
    # plot the 3 sliced images (raw images in counts)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    im = axes[0].imshow(im_core[core_slice] * calib_params['exp_min'] * 10**(-calib_params['atten_min']/10))
    axes[0].set_title('core slice, calib, counts')
    fig.colorbar(im, ax=axes[0])
    im = axes[1].imshow(im_speckle[speckle_slice] * calib_params['exp_max'] * 10**(-calib_params['atten_max']/10))
    axes[1].set_title('speckle slice, calib, counts')
    fig.colorbar(im, ax=axes[1])
    im = axes[2].imshow(im_dh[speckle_slice] * calib_params['exp_dh'] * 10**(-calib_params['atten_max']/10))
    axes[2].set_title('speckle slice, DH, counts')
    fig.colorbar(im, ax=axes[2])
    
    # plot the 3 sliced images (aperture convolved, counts per second)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    axes[0].imshow(calib_proc['core_calib_cps_apconv'][core_slice])
    axes[0].set_title('core slice, calib, ap-convolved, cps')
    axes[1].imshow(calib_proc['speckle_calib_cps_apconv'][speckle_slice])
    axes[1].set_title('speckle slice, calib, ap-convolved, cps')
    axes[2].imshow(calib_proc['speckle_dh_cps_apconv'][speckle_slice])
    axes[2].set_title('speckle slice, DH, ap-convolved, cps')
    
    # cross-checks
    core_cps_dh = calib_proc['core_cps_dh']
    speckle_contrast = calib_proc['speckle_dh_cps_apconv'][speckle_slice].max() / core_cps_dh
    core_contrast = calib_proc['core_calib_cps_apconv'][core_slice].max()  / calib_proc['atten'] / core_cps_dh
    
    # print out the attenuation factor, the core normalization, the core contrast (better = 1), and the speckle contrast
    print(f'Attenuation factor: {calib_proc["atten"]}')
    print(f'Core CPS: {calib_proc["core_cps_dh"]}')
    print(f'Speckle contrast: {speckle_contrast}')
    print(f'Core contrast: {core_contrast}')
    
class CameraStream(ImageStream):
    
    def __init__(self, name, client, device, fiberatten, sliceyx=None):
        super().__init__(name)
        
        self.client = client
        self.device = device
        self.fiberatten = fiberatten
        if sliceyx is None:
            sliceyx = (slice(None,None), slice(None,None))
        self.sliceyx = sliceyx
        self.exp_time = None
        self.atten = None
        
    def set_exposure_time(self, exp, wait=None):
        client[f'{self.device}.exptime.target'] = exp
        #client[f'{self.device}.exptime.exptime'] = exp
        if wait is not None:
            sleep(wait)
        # update tracked exposure tiem
        self.get_exposure_time()
        
    def set_attenuation(self, atten, wait=None):
        client[f'{self.fiberatten}.atten.target'] = atten
        if wait is not None:
            sleep(wait)
        # update tracked attenuation
        self.atten = 10**(-atten/10.)
        
    def set_gain(self, gain, wait=None):
        client[f'{self.device}.emgain.target'] = gain
        if wait is not None:
            sleep(wait)
        # update tracked attenuation
        self.gain = gain #10**(-atten/10.)
        
    def get_exposure_time(self):
        t = client[f'{self.device}.exptime.target']
        self.exp_time = t
        return t
    
    #def grab_latest(self):
    #    # grab_many and grab_after both call grab_latest internally
    #    out = super().grab_latest()
    #    return downscale_local_mean(out.astype(float), (2,2)) / self.exp_time
    
    def grab_latest(self):
        # grab_many and grab_after both call grab_latest internally
        out = super().grab_latest()[self.sliceyx]
        return out.astype(float) / self.exp_time / self.atten
    
    #def grab_many(self, n, cnt0_min=None):
    #    out = super().grab_many(n, cnt0_min=cnt0_min)
    #    return np.asarray(out, dtype=float) / self.exp_time
    
    #def grab_after(self, n, nwait):
    #    out = super().grab_after(n, nwait)
    #    return np.asarray(out, dtype=float) / self.exp_time
    
def move_relative(client, device, val):
    client[f'{device}.target'] = client[f'{device}.current'] + val
    
def move_linear(client, val):
    client['stagelinear.position.target'] = val
    
def move_linear_preset(client, preset):
    #client[f'stagelinear.presetName.{preset}'] = SwitchState.ON
    instrument.indi_send_and_wait(client,
                                  {f'stagelinear.presetName.{preset}' : SwitchState0.ON},
                                  timeout=30)
    
def take_dark(client, camstream, nimages, restore=True):
    move_linear_preset(client, 'block_in')
    sleep(5) # not sure
    dark = np.mean(camstream.grab_many(nimages), axis=0)
    if restore:
        move_linear_preset(client, 'block_out')
    return dark

def take_bg(n=100, nwait=2):
    move_linear_preset(client0, 'block_in')
    imbg = np.mean(camstream.grab_after(n, nwait=nwait), axis=0)
    move_linear_preset(client0, 'block_out')
    return imbg

def take_bg_atten(n=100, nwait=2):
    atten0 = np.abs(-np.log10(camstream.atten) * 10) # force positive for the -0.0 case
    camstream.set_attenuation(60, wait=10)
    move_linear_preset(client0, 'block_in')
    imbg = np.mean(camstream.grab_after(n, nwait=nwait), axis=0) * 10**(-(60-atten0)/10) #???
    move_linear_preset(client0, 'block_out')
    camstream.set_attenuation(atten0, wait=0)
    return imbg

def get_mean_contrast(imdh, imbg, mask, norm_cps, rejectnegative=True):
    bg_sub = imdh-imbg
    if rejectnegative:
        bg_sub[bg_sub < 0] = np.nan
    return np.nanmean(bg_sub[mask]) / norm_cps

def save_dict_to_h5(filename, mydict):
    dd.io.save(filename, mydict)
    
def load_dict_from_h5(filename):
    return dd.io.load(filename)