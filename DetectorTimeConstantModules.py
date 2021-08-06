######################################Detector Time Constant Modules################################

def lognorm(x, mu, sigma):
    import numpy as np
    return 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2. / (2 * sigma**2.)) 

def gamma(x, theta=2.5, k=2):
    #mu = k*theta
    #var = k*theta**2.
    import numpy as np
    import math
    return (x**(k-1) * np.exp(- x / theta)) / (theta**k * math.gamma(k))


def sample_det_tau(det_dict, max_x=50, theta=2.5, k=2):
    import numpy as np
    
    y_max = gamma(theta, theta=theta, k=k)
    for freq in det_dict.keys():
        #assert('tau' not in det_dict[freq][0].keys())
        for det in det_dict[freq].keys():
            
            #keep sampling until acceptance
            sample_again = True
            while(sample_again):
            
                #step1 sample x
                x = max_x * np.random.rand()

                #step2 calculate f(x)
                fx = gamma(x, theta=theta, k=k)

                #step3 sample y
                y = y_max * np.random.rand()

                #acceptance/rejection
                if y > fx:
                    pass
                elif x < 1:
                    pass
                else:
                    det_dict[freq][det]['tau'] = x
                    sample_again = False  
    
    return det_dict

def gen_scanx_dir_map(det_tau, obs_freq, scan_sp=3., N=1024, pixel_size=0.25/60., D_aper=5.):
    import scipy.constants as constants
    import numpy as np
    
    #relavant params: observing freq, scan speed in deg/s, N resolution of map, pixel size in deg, diameter of main aperture, det tau
    ell_fac = 100. / 1. #conversion factor from deg to ell
    obs_lam = constants.c / (obs_freq*10**9)
    diff_lim = 1.22*(obs_lam / D_aper) * (180. / constants.pi) * ell_fac
    ell_cutoff = (scan_sp * det_tau) * ell_fac
    
    ell_max = (N*pixel_size)*ell_fac
    ell = np.linspace(0, ell_max, pixel_size)
    
    #create empty map
    map_scan_fft = np.zeros((N,N))
    
    #fill each row with lpf TF
    for i in np.arange(N):
        map_scan_fft[:][i] = lpf_tf_amp(ell, scan_sp, det_tau)
    
    #return whole map
    
    return map_scan_fft

def lpf_tf(ell, scan_sp, det_tau):
    return 1 / ( 1 + (1j * ell / (100/(scan_sp*det_tau)) ) )

def lpf_tf_amp(ell, scan_sp, det_tau):
    import numpy as np
    return 1 / np.sqrt( 1 + (ell**2.*(scan_sp*det_tau)**2)/1e4 )

def convolve_scan_direction(Map, map_scan_fft):
    import numpy as np
    Map_fft = np.fft.fft2(np.fft.fftshift(Map))
    #convolved_map = np.fft.fftshift(np.real(Map_fft * map_scan_fft))
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(Map_fft * map_scan_fft)))
    return convolved_map

 ###############################  ###############################  ###############################