import numpy as np
import scipy.stats as stats



def gauss1D_cart(x, mu=0.0, sigma=1.0):
    """gauss1D_cart

    gauss1D_cart takes a 1D array x, a mean and standard deviation,
    and produces a gaussian with given parameters, with a peak of height 1.

    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the gauss
    mu : float, optional
        mean/mode of gaussian (the default is 0.0)
    sigma : float, optional
        standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray
        gaussian values at x
    """

    return np.exp(-((x-mu)**2)/(2*sigma**2)).astype('float32')



def gauss1D_log(x, mu=0.0, sigma=1.0):
    """gauss1D_log

    gauss1D_log takes a 1D array x, a mean and standard deviation,
    and produces a pRF with given parameters with the distance between mean and x log-scaled 

    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the gauss
    mu : float, optional
        mean/mode of gaussian (the default is 0.0)
    sigma : float, optional
        standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray
        gaussian values at log(x)
    """

    return np.exp(-(np.log(x/mu)**2)/(2*sigma**2)).astype('float32')



def vonMises1D(x, mu=0.0, kappa=1.0):
    """vonMises1D

    vonMises1D takes a 1D array x, a mean and kappa (inverse of standard deviation),
    and produces a von Mises pRF with given parameters. This shape can be thought of 
    as a circular gaussian shape. Used for orientation or motion direction pRFs, 
    for instance.

    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the von Mises. 
        Assumed to be in the range (0, 2*np.pi)
    mu : float, optional
        mean/mode of von Mises (the default is 0.0)
    kappa : float, optional
        dispersion coefficient of the von Mises, 
        akin to invers of standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray
        von Mises values at x, peak has y-value of 1
    """
    vm = stats.vonmises.pdf(x-mu, kappa)

    return vm / np.max(vm)



def gauss2D_iso_cart(x, y, mu=(0.0, 0.0), sigma=1.0, normalize_RFs=False):
    """gauss2D_iso_cart

    gauss2D_iso_cart takes two-dimensional arrays x and y, containing
    the x and y coordinates at which to evaluate the 2D isotropic gaussian 
    function, with a given sigma, and returns a 2D array of Z values.

    Parameters
    ----------
    x : numpy.ndarray, 2D or flattened by masking
        2D, containing x coordinates
    y : numpy.ndarray, 2D or flattened by masking
        2D, containing y coordinates
    mu : tuple, optional
        mean, 2D coordinates of mean/mode of gauss (the default is (0.0,0.0))
    sigma : float, optional
        standard deviation of gauss (the default is 1.0)

    Returns 
    -------
    numpy.ndarray, 2D or flattened by masking
        gaussian values evaluated at (x,y)
    """
    if normalize_RFs:
        return (np.exp(-((x-mu[0])**2 + (y-mu[1])**2)/(2*sigma**2)) /(2*np.pi*sigma**2)).astype('float32')
    else:
        return (np.exp(-((x-mu[0])**2 + (y-mu[1])**2)/(2*sigma**2))).astype('float32')



def gauss2D_rot_cart(x, y, mu=(0.0, 0.0), sigma=1.0, theta=0.0, ar=1.0):
    """gauss2D_rot_cart

    gauss2D_rot_cart takes two-dimensional arrays x and y, containing
    the x and y coordinates at which to evaluate the 2D non-isotropic gaussian 
    function, with a given sigma, angle of rotation theta, and aspect ratio ar.
    it returns a 2D array of Z values. Default is an isotropic gauss.

    Parameters
    ----------
    x : numpy.ndarray, 2D
        2D, containing x coordinates or flattened by masking
    y : numpy.ndarray, 2D
        2D, containing y coordinates or flattened by masking
    mu : tuple, optional
        mean, 2D coordinates of mean/mode of gauss (the default is (0.0,0.0))
    sigma : float, optional
        standard deviation of gauss (the default is 1.0)
    theta : float, optional
        angle of rotation of gauss (the default is 0.0)   
    ar : float, optional
        aspect ratio of gauss, multiplies the rotated y parameters (the default is 1.0)

    Returns
    -------
    numpy.ndarray, 2D or flattened by masking
        gaussian values evaluated at (x,y) 
    """
    xr = (x-mu[0]) * np.cos(theta) + (y-mu[1]) * np.sin(theta)
    yr = -(x-mu[0]) * np.sin(theta) + (y-mu[1]) * np.cos(theta)

    return np.exp(-(xr**2 + ar**2 * yr**2)/(2*sigma**2))



def gauss2D_logpolar(ecc, polar, mu=(1.0, 0.0), sigma=1.0, kappa=1.0):
    """gauss2D_logpolar

    gauss2D_logpolar takes two-dimensional arrays ecc and polar, containing
    the eccentricity and polar angle coordinates at which to evaluate the 2D gaussian, 
    which in this case is a von Mises in the polar angle direction, and a log gauss 
    in the eccentricity dimension, and returns a 2D array of Z values.
    We recommend entering the ecc and polar angles ordered as x and y for easy
    visualization.

    Parameters
    ----------
    ecc : numpy.ndarray, 2D or flattened by masking
        2D, containing eccentricity
    polar : numpy.ndarray, 2D or flattened by masking
        2D, containing polar angle coordinates (0, 2*np.pi)
    mu : tuple, optional
        mean, 2D coordinates of mean/mode of gauss (ecc) and von Mises (polar) (the default is (0.0,0.0))
    sigma : float, optional
        standard deviation of gauss (the default is 1.0)
    kappa : float, optional
        dispersion coefficient of the von Mises, 
        akin to inverse of standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray, 2D or flattened by masking
        values evaluated at (ecc, polar), peak has y-value of 1.
    """
    ecc_gauss = np.exp(-(np.log(ecc/mu[0])**2)/(2*sigma**2))
    polar_von_mises = stats.vonmises.pdf(polar-mu[1], kappa)
    polar_von_mises /= np.max(polar_von_mises)
    logpolar_Z = ecc_gauss * polar_von_mises

    return logpolar_Z / np.max(logpolar_Z)

# ************************************************************************************************************
# CSenF functions
def csenf_exponential(log_SF_grid, CON_S_grid, width_r, SFp, CSp, width_l, **kwargs):
    '''
    Python version written by Marcus Daghlian, translated from matlab original (credit Carlien Roelofzen) 
    Note now we generally fit CRF as well 

    Takes a set of parameters determining the CSF (& CRF), and projects these onto a matrix representing log spatial frequency and contrast sensitivity
    Conceptually akin to a receptive field, but in SF-contrast space, not visual (x,y) space
    
    inputs:
    -------
    
    log_SF_grid        : grid of log10 SF values 
    CON_S_grid         : grid of 100/contrast sensitivity values 
    ***
    width_r     : width of CSF function, curvature of the parabolic
                function (larger values mean narrower function)
                width is the right side of the curve (width_right)                
    SFp        : spatial frequency with peak sensitivity  
    CSp       : maximale contrast at SFp
    width_l    : width of the left side of the CSF curve,

    Optional:
    width_l_type : 'default', 'ratio' (=width_r*0.5308)
                'default' just accepts the width_l values input
                If you want to fix the values to (=0.4480), it is better to do this outside the function 
                'ratio' overwrites width_l to be a function of width_r. (=0.5308.*widht_right)
    edge_type   : 'CRF' (default),'gauss', 'binary' 
    crf_exp     : exponent for the smooth rf (CRF) default = 1.                 
    scaling_factor : scaling factor for CRF, default = 1
    return_curve    : bool, return the curve as well (default false)
    '''
    # How many RFs are we making?
    # if not isinstance(width_r, np.ndarray)     :
    if len(width_r) == 1:
        n_RFs = 1
    else:
        n_RFs = width_r.shape[0] # all RF parameters should be the same length

    # Setup kwargs
    width_l_type = kwargs.get('width_l_type', 'default')
    if width_l_type == 'ratio': 
        print('CHANGING WIDTH L')
        width_l = width_r * 0.4480
    edge_type = kwargs.get('edge_type', 'CRF')
    crf_exp = kwargs.get('crf_exp', 1)                  # 1
    scaling_factor = kwargs.get('scaling_factor', 1)    # 1
    return_curve = kwargs.get('return_curve', False)
    # CONVERT SFp and CSp
    log_SFp = np.log10(SFp)
    log_CSp = np.log10(CSp)
    log_sfs_gr = np.moveaxis(np.tile(log_SF_grid, (n_RFs, 1,1)), 0, -1)
    con_s_gr = np.moveaxis(np.tile(CON_S_grid, (n_RFs, 1,1)), 0, -1) 

    # Reshape RF parameters 

    width_r     = np.reshape(width_r, (1,1,n_RFs))  
    log_SFp     = np.reshape(log_SFp, (1,1,n_RFs))
    log_CSp    = np.reshape(log_CSp, (1,1,n_RFs))
    width_l     = np.reshape(width_l, (1,1,n_RFs))
    crf_exp     = np.reshape(crf_exp, (1,1,n_RFs))    
    
    # Split the stimulus space into L & R of the SFp
    id_SF_left  = log_sfs_gr <  log_SFp
    id_SF_right = log_sfs_gr >= log_SFp
    # Create the curves    
    L_curve = L_curve = 10**(log_CSp - ((log_sfs_gr-log_SFp)**2) * (width_l**2))
    R_curve = R_curve = 10**(log_CSp - ((log_sfs_gr-log_SFp)**2) * (width_r**2))
    csf_curve = np.zeros_like(L_curve)
    csf_curve[id_SF_left] = L_curve[id_SF_left]
    csf_curve[id_SF_right] = R_curve[id_SF_right]
    # edge_type = 'binary'
    if edge_type=='CRF':
        # Smooth Contrast Response Function (CRF) 
        # Standard Naka-Rushton function, as used by Wietske Zuiderbaan and Boynton (1999).
        # >> R(C) = C^q / (C^q + Q^q) 
        # >> Q determines where R=0.5 (we use the csf_curve)
        # >> q determines the slope (see crf_exp)
        # Note we want contrasts, not 100/contrast, so we need to do this... 
        con_gr = 100/con_s_gr       # from contrast sensitivity -> contrast
        c_curve = 100/csf_curve     # from contrast sensitivity -> contrast        
        # c_curve[np.isnan(c_curve)] = np.inf     # dividing by 0! dirty fix here        
        csf_rfs = scaling_factor * ((con_gr**crf_exp) / (con_gr**crf_exp + c_curve**crf_exp))


    elif edge_type=='binary':
        # Simple binary version. Contrast level below the curve is 1, anything above it is 0
        csf_rfs = con_s_gr<=csf_curve

    # Reshape...
    csf_rfs = np.moveaxis(csf_rfs, -1, 0)

    if return_curve:
        return csf_rfs, csf_curve[0,:,:]

    return csf_rfs