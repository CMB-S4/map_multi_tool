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

def generate_2d_gaussian_beam(N=1024, pixel_size=0.25, sigma_x=1.5, sigma_y=1.5):
    import numpy as np
    
    L = N*pixel_size    #size of map in arcmin

    #define grid
    x_1d = (np.arange(N)/N - 0.5)*L
    y_1d = np.copy(x_1d)
    x,y = np.meshgrid(x_1d,y_1d)
    
    beam = np.exp(-0.5*((x/sigma_x)**2. + (y/sigma_y)**2.))
    return beam

def generate_2d_residual_beam(sigma_x, sigma_y, sigma_x_prime, sigma_y_prime, N=1024, pixel_size=0.25):
    import numpy as np
    
    L = N*pixel_size    #size of map in arcmin

    #define grid
    x_1d = (np.arange(N)/N - 0.5)*L
    y_1d = np.copy(x_1d)
    x,y = np.meshgrid(x_1d,y_1d)
    
    beam_residual = np.exp(-0.125*((x/sigma_x)**2. + (y/sigma_y)**2. - (x/sigma_x_prime)**2. - (y/sigma_y_prime)**2.))
    
    return beam_residual

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
    return 1 / np.sqrt( 1 + (ell**2.*(scan_sp*det_tau)**2.)/1e4 )

def convolve_scan_direction(Map, map_scan_fft):
    import numpy as np
    Map_fft = np.fft.fft2(np.fft.fftshift(Map))
    #convolved_map = np.fft.fftshift(np.real(Map_fft * map_scan_fft))
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(Map_fft * map_scan_fft)))
    return convolved_map

 ###############################  ###############################  ###############################

#Fit Gauss to a Gaussian
def gaus(x,a,sig):
    import numpy as np
    return a*np.exp(-(x)**2/(2.*sig**2))

# Returns the fractional window function uncertainty as a function of
# ell=np.linspace(0,50000,50001)
def get_tc_uncertainty(fwhm,f3db):
    import numpy as np
    from scipy.optimize import curve_fit
    
    #parameters
    el= 45. #scan elevation in degrees
    fscan= 1.#scan speed of the telescope deg/s
    fsky=fscan*np.cos(el*np.pi/180.) #deg/s on sky scan speed
    #frequency space Gaussian beam
    sigma=fwhm/(60.*2.*fsky*np.sqrt(2.*np.log(2)))
    t_range=100000.
    t_points=10000000
    t_freq=np.linspace(-t_range,t_range,t_points)
    sample_rate=(t_points/(2.*t_range))
    gauss_2=np.exp(-2.*np.pi**2*t_freq**2*sigma**2)

    pshift=np.zeros(len(f3db))
    FWHM_new=np.zeros(len(f3db))
    for ii in range(len(f3db)):

            #Make Lowpass Filter (and apply to negative side--this is just the scan direction)
            h=(1.-1j*(-t_freq/f3db[ii]))/(1.+np.square(-t_freq/f3db[ii]))
            filtered_2=np.copy(gauss_2)*h
            filtered=np.copy(filtered_2)

            #Take FFT of Convoluted Beam
            gffted=np.abs(np.fft.fft(gauss_2))
            ffted=np.fft.fft(filtered)
            rffted=np.absolute(ffted)
            #make position array of the same length
            pos1=np.linspace(0,len(rffted)-1,len(rffted))
            #normalize to one and scale by sample rate and convert into degrees
            pos2=(pos1/len(rffted))*sample_rate*fsky

            #put the plot back together
            #First section of fft
            fftr1=rffted[0:int(len(rffted)/2)]
            #print pointing offset
            pshift[ii]=pos2[np.argmax(fftr1)]*60.
            #print f3db[ii], 'Hz', pshift[ii], 'arcmin'
            #Find x value where value is half
            x=np.copy(pos2[0:int(len(pos2)/2)])
            yf=np.copy(fftr1)
            y=yf/np.max(yf)
            xp=np.fliplr([x])[0]
            yp=np.fliplr([y])[0]
            half=np.interp([0.5],yp,xp)
            FWHM_new[ii]=(half[0]*60.-pshift[ii])*2.
            #print "Beam FWHM: ",FWHM_new[ii], "arcmin"
            #print "Change in FWHM: ",FWHM_new[ii]-fwhm, "arcmin"
            
    # calculate error in pointing from time constant uncertainty
    p_10=pshift[1]-pshift[0]
    #print "Pointing error: ", p_10*60., "arcsec"
    print( "Pointing error: " + str(p_10*60.) + "arcsec")
    #Take units in arcminutes
    fwhm_shift=np.zeros(len(f3db))
    for jj in range(len(f3db)):
      #print '%0.2f Hz:'%f3db[jj]
      angle=np.linspace(-500,500,100000)
      poff=pshift[jj]
      sigma2=FWHM_new[jj]/(2.*np.sqrt(2.*np.log(2)))
      a=1./(sigma2*np.sqrt(2*np.pi))
      top1=-0.5*((angle-poff)**2/sigma2**2)
      top2=-0.5*((angle+poff)**2/sigma2**2)
      #These are the two gaussians
      gauss_a=a*np.exp(top1)
      gauss_b=a*np.exp(top2)

      #Make the combined beam
      gauss=gauss_a+gauss_b

      #Find FWHM and thus the best guess of sigma (assume centered around zero)
      aa=np.copy(angle)
      yyf=np.copy(gauss)
      yy=yyf/np.max(yyf)
      half=np.interp([0.5],yy[0:int(len(angle)/2)],aa[0:int(len(angle)/2)])
      fwhm_fit=np.absolute(2*half[0])
      #print 'Guess: ',fwhm_fit
      sigfit=fwhm_fit/(2.*np.sqrt(2.*np.log(2)))

      popt,pcov = curve_fit(gaus,angle,gauss,p0=[1,sigfit])
      fwhm_shift[jj]=popt[1]*(2.*np.sqrt(2.*np.log(2)))
      #print 'FWHM Fit: ',fwhm_shift[jj], "arcmin"
      #print 'Change in FWHM: ',popt[1]*(2.*np.sqrt(2.*np.log(2)))-fwhm
      #take these and look at their Gaussian window functions
    l=np.linspace(0,50000,50001)
    sigma_nt=(fwhm/60.)*(np.pi/180.)/np.sqrt(8*np.log(2))
    #Gaussian window function, no time constant
    bl=np.exp(-l*(l+1)*sigma_nt**2)
    #with base time constant
    sigma_tc=(fwhm_shift[0]/60.)*(np.pi/180.)/np.sqrt(8*np.log(2))
    blm_orig=np.exp(-l*(l+1)*sigma_tc**2)
    #base time constant + err
    sigma_mod=(fwhm_shift[1]/60.)*(np.pi/180.)/np.sqrt(8*np.log(2))
    blm=np.exp(-l*(l+1)*sigma_mod**2)
    #pct_diff
    pct_diff_tc=(blm_orig-blm_orig)/bl
    pct_diff=(blm-blm_orig)/bl
    return pct_diff

#returns the percent difference in window function
def get_window_function(fwhm, f3db_b):
    import numpy as np
#    fwhm=2.1 #beam FWHM in arcmin
#    f3db_b=144.
    f3db=[f3db_b,(f3db_b-0.3*f3db_b)] #30% uncertainty
    pct_diff=get_tc_uncertainty(fwhm,f3db)
    print( "fractional difference at ell=5000: " + str(pct_diff[5001]))
    
    return pct_diff