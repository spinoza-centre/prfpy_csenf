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
def nCSF_response_grid(SF_list, CON_list, width_r, SFp, CSp, width_l, crf_exp, **kwargs):
    '''nCSF_response
    Same as nCSF_response, but sometimes we want to use a grid
    i.e., get responses across SF-contrast space    
    '''
    SF_grid, CON_grid = np.meshgrid(SF_list, CON_list)
    ncsf_grid_shape = SF_grid.shape
    ncsf_response = nCSF_response(
        SF_grid.flatten(), CON_grid.flatten(), width_r, SFp, CSp, width_l, crf_exp, **kwargs        
    )
    ncsf_response = ncsf_response.reshape(len(width_r), ncsf_grid_shape[0], ncsf_grid_shape[1],)
    return ncsf_response

def nCSF_response(SF_seq, CON_seq, width_r, SFp, CSp, width_l, crf_exp, **kwargs):
    '''nCSF_response
    Response of nCSF models with parameters width_r,SFp,CSp,width_l,crf_exp
    To SF and contrast pairs at each time point in the sequence
    Unconvolved with the HRF...    
    '''
    edge_type = kwargs.get('edge_type', 'CRF') # default CRF, other option is binary    
    if not isinstance(width_r, np.ndarray):
        width_r = np.atleast_1d(np.array(width_r))
        SFp     = np.atleast_1d(np.array(SFp))
        CSp     = np.atleast_1d(np.array(CSp))
        width_l = np.atleast_1d(np.array(width_l))    
        crf_exp = np.atleast_1d(np.array(crf_exp))

    csenf_values = asymmetric_parabolic_CSF(SF_seq, width_r, SFp, CSp, width_l)

    # Reshape nCSF parameters 
    crf_exp = crf_exp.reshape(-1,1)#,1)
    # Reshape stimulus sequence
    CON_seq = CON_seq.reshape(1,-1)#,1)
    # Apply the edge method 
    ncsf_response = nCSF_apply_edge(
        csenf_values=csenf_values,
        crf_exp = crf_exp, 
        CON_seq=CON_seq,
        edge_type=edge_type,
    )

    return ncsf_response

def nCSF_apply_edge(csenf_values, crf_exp, CON_seq, edge_type):
    ''' Given the CSF and the contrasts presented, determine the response
    > binary edge: all contrasts below sensitivity = 1, above = 0
    > CRF naka rushton function applied
    > ...
    '''
    # convert from contrast sensitivity to contrast threshold...
    cthresh_values = 100/csenf_values
    # Now we have the csenf_values at each SF    
    if edge_type=='CRF':
        # Smooth Contrast Response Function (CRF) 
        # Simplified Naka-Rushton function
        # >> R(C) = C^q / (C^q + Q^q) 
        # >> Q determines where R=0.5 (we use the csf_curve)
        # >> q determines the slope (see crf_exp)    
        ncsf_response = ((CON_seq**crf_exp) / (CON_seq**crf_exp + cthresh_values**crf_exp))
    elif edge_type=='binary':
        # Everything below csenf is 1, above = 0
        ncsf_response = (100/CON_seq)<=csenf_values
    elif edge_type=='straight':        
        # Straight line with 0=0, and 0.5=Q 
        ncsf_response = (CON_seq / (cthresh_values*2)) 
    elif edge_type=='css_compare':
        # Straight line with 0=0, and 0.5=Q 
        # Then exponent
        ncsf_response = (CON_seq / (cthresh_values*2))  ** crf_exp
    elif edge_type=='bound_slope':
        # 0.5 = Q, slope is determined by crf_exp
        # All values <0 =0, >1 = 1...
        ncsf_response = ((CON_seq*crf_exp) / cthresh_values) - crf_exp + 0.5            
        ncsf_response[ncsf_response<0] = 0
        ncsf_response[ncsf_response>1] = 1        

    return ncsf_response

def asymmetric_parabolic_CSF(SF_seq, width_r, SFp, CSp, width_l, **kwargs):
    '''asymmetric_parabolic_CSF
    The CSF component is parameterized as in Chung & Legge 2016 (DOI:10.1167/ iovs.15-18084) 
    > parameters: width_r, SFp, CSp, width_l

    
    Parameters:
    -------
    SF_seq : numpy.ndarray
        SF values 
    CON_S_grid : numpy.ndarray
        Grid of 100/contrast values
    width_r : numpy.ndarray or float
        Width of the CSF function, curvature of the parabolic function (larger values mean narrower function)
    SFp : float
        Spatial frequency with peak sensitivity
    CSp : float
        Maximal contrast at SFp
    width_l : numpy.ndarray or float
        Width of the left side of the CSF curve
    
    Returns:
    -------
    csf_curve : numpy.ndarray
        Contrast sensitivity at each of the SFs in SF_list

    '''

    if not isinstance(width_r, np.ndarray):
        width_r = np.atleast_1d(np.array(width_r))
        SFp     = np.atleast_1d(np.array(SFp))
        CSp     = np.atleast_1d(np.array(CSp))
        width_l = np.atleast_1d(np.array(width_l))

    # CONVERT SFp and CSp and SFs to log10 versions
    log_SF_seq  = np.log10(SF_seq)
    log_SFp = np.log10(SFp)
    log_CSp = np.log10(CSp)
    
    # Reshape CSF parameters 
    width_r     = width_r.reshape(-1,1)
    log_SFp     = log_SFp.reshape(-1,1)
    log_CSp     = log_CSp.reshape(-1,1)
    width_l     = width_l.reshape(-1,1)
    
    # Reshape stimulus (orthogonal to the parameters)
    log_SF_seq = log_SF_seq.reshape(1,-1)
    # Split the stimulus space into L & R of the SFp
    id_SF_left  = log_SF_seq <  log_SFp
    id_SF_right = log_SF_seq >= log_SFp

    # Create the curves    
    L_curve = 10**(log_CSp - ((log_SF_seq-log_SFp)**2) * (width_l**2))
    R_curve = 10**(log_CSp - ((log_SF_seq-log_SFp)**2) * (width_r**2))
    csf_curve = np.zeros_like(L_curve)
    csf_curve[id_SF_left] = L_curve[id_SF_left]
    csf_curve[id_SF_right] = R_curve[id_SF_right]

    return csf_curve
