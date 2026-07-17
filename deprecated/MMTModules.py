#############################################Simulation Modules###########################################
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.random as random
#import pandas as pd
import math

def calculate_crosstalk(det_dict, coupling_dict, freqs, pixel_size, perc_corr, N, beam_fwhm, sky_decomp, TtoP_suppress, delta_ell, ell_max, choose_normalization, unconvolved_beams=None):
    '''
    calculates and returns the 3x3 beam coupling matrix given a crosstalk matrix (coupling_dict) and a detector layout specified in det_dict. Then convolves the pixel beam matrix components with the instrument beam and plots the beam coupling matrix in real space on a log scale.
    
    
    Inputs:
    ****************************************************************************
    det_dict [dictionary]: dictionary containing identifying information for the detectors in the focal plane
    coupling_dict [dictionary]: crosstalk matrix defining the detector to detector coupling
    freqs [2D array]: list of band frequencies being considered for the analysis
    pixel_size [float]: size of each pixel in arcmin
    perc_corr [float]: percent to which the crosstalk signal is correlated to the carrier signal
    N [int]: number of pixels along a map edge
    beam_fwhm [float]: 3dB point of gaussian beam
    sky_decomp [1D array]: a list of either the linear decomposition of the sky signal in IQU space or CMB realizations of I, Q, and U maps without instrument and observation effects
    TtoP_suppress [boolean]: sets the central value of T->P coupling maps to zero in where it otherwise may not be due to an imbalance of pair-differenced detectors
    delta_ell [int]: bin size of power spectra in ell
    ell_max [int]: maximum ell of power spectra calculation
    choose_normalization [string or 0]: if string it will be a key of the autospectra (TT for example), this will normalize every leakage spectra to the peak of this particular autospectra. If 0 is entered, each spectra is peak normalized.
    unconvolved_beams [dictionary]: *Deprecated do not use*
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    convolved_coupled_beams [dictionary]: 3x3 beam coupling matrix
    ****************************************************************************
    '''
    
    #frequency under question
    freq1 = freqs[0]
    #frequency correlated to freq1
    freq2 = freqs[1]
    
    #run a simulation
    beam_matrix = calculate_beam_matrix(det_dict, coupling_dict, freq1, freq2, pixel_size, perc_corr, N, TtoP_suppress = TtoP_suppress)
    II_total = beam_matrix['II']
    IQ_total = beam_matrix['IQ']
    IU_total = beam_matrix['IU']
    QI_total = beam_matrix['QI']
    QQ_total = beam_matrix['QQ']
    QU_total = beam_matrix['QU']
    UI_total = beam_matrix['UI']
    UQ_total = beam_matrix['UQ']
    UU_total = beam_matrix['UU']

    #central values of beam maps
    print('central beam values II:' + str(II_total[int(N/2),int(N/2)]) + ', QQ:' + str(QQ_total[int(N/2),int(N/2)]) + ', UU:' + str(UU_total[int(N/2),int(N/2)]))
    print('IQ: ' + str(IQ_total[int(N/2),int(N/2)]))
    print('IU: ' + str(IU_total[int(N/2),int(N/2)]))
    print('QI: ' + str(QI_total[int(N/2),int(N/2)]))
    print('QU: ' + str(QU_total[int(N/2),int(N/2)]))
    print('UI: ' + str(UI_total[int(N/2),int(N/2)]))
    print('UQ: ' + str(UQ_total[int(N/2),int(N/2)]))

    #convolve with instrument beam and organize into dictionaries
    coupled_beams = {}
    inst_beam_1 = offset_2d_gaussian_beam(N, pixel_size, beam_fwhm,0,0)
    coupled_beams_keys = ['II','QI','UI','IQ','QQ','UQ','IU','QU','UU']
    beams_list = [II_total, QI_total, UI_total, IQ_total, QQ_total, UQ_total, IU_total, QU_total, UU_total]
    convolved_coupled_beams = {}
    for i in range(len(coupled_beams_keys)):
        coupled_beams[coupled_beams_keys[i]] = beams_list[i]

    for i in range(len(coupled_beams_keys)):
        convolved_coupled_beams[coupled_beams_keys[i]] = convolve_pixel_instrument(coupled_beams[coupled_beams_keys[i]],inst_beam_1)

    #Plot beam maps in 3x3 matrix
    fig, ax = plt.subplots(3,3, figsize=(20,20))
    
    #tiny offset to avoid numerical errors in log space
    eta = 0.0000001
    
    #make colormap relative to II beam
    vmin = 10.*np.log(np.min(II_total) + eta)
    vmax = 10.*np.log(np.max(II_total) + eta)

    #II
    ax[0,0].imshow(10. * np.log(np.abs(convolved_coupled_beams['II']) + eta),vmin=vmin,vmax=vmax)
    ax[0,0].set_xlabel('X Pixel')
    ax[0,0].set_ylabel('Y Pixel')
    ax[0,0].set_title('II Beam')
    
    #IQ
    ax[0,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['IQ']) + eta),vmin=vmin,vmax=vmax)
    ax[0,1].set_xlabel('X Pixel')
    ax[0,1].set_ylabel('Y Pixel')
    ax[0,1].set_title('IQ Beam')

    #IU
    ax[0,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['IU']) + eta),vmin=vmin,vmax=vmax)
    ax[0,2].set_xlabel('X Pixel')
    ax[0,2].set_ylabel('Y Pixel')
    ax[0,2].set_title('IU beam')

    #QI
    ax[1,0].imshow(10.*np.log(np.abs(convolved_coupled_beams['QI']) + eta),vmin=vmin,vmax=vmax)
    ax[1,0].set_xlabel('X Pixel')
    ax[1,0].set_ylabel('Y Pixel')
    ax[1,0].set_title('QI beam')

    #QQ
    ax[1,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['QQ']) + eta),vmin=vmin,vmax=vmax)
    ax[1,1].set_xlabel('X Pixel')
    ax[1,1].set_ylabel('Y Pixel')
    ax[1,1].set_title('QQ beam')

    #QU
    ax[1,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['QU']) + eta),vmin=vmin,vmax=vmax)
    ax[1,2].set_xlabel('X Pixel')
    ax[1,2].set_ylabel('Y Pixel')
    ax[1,2].set_title('QU beam')

    #UI
    ax[2,0].imshow(10. * np.log(np.abs(convolved_coupled_beams['UI']) + eta),vmin=vmin,vmax=vmax)
    ax[2,0].set_xlabel('X Pixel')
    ax[2,0].set_ylabel('Y Pixel')
    ax[2,0].set_title('UI beam')

    #UQ
    ax[2,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['UQ']) + eta),vmin=vmin,vmax=vmax)
    ax[2,1].set_xlabel('X Pixel')
    ax[2,1].set_ylabel('Y Pixel')
    ax[2,1].set_title('UQ beam')

    #UU
    ax[2,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['UU']) + eta),vmin=vmin,vmax=vmax)
    ax[2,2].set_xlabel('X Pixel')
    ax[2,2].set_ylabel('Y Pixel')
    ax[2,2].set_title('UU beam')
    plt.savefig("Beam Matrix.png")
    plt.show()

    return convolved_coupled_beams

def get_leakage_spectra(convolved_coupled_beams, pixel_size, N, beam_fwhm, sky_decomp, delta_ell, ell_max, choose_normalization, plots=False):
    
    '''
    from a given beam coupling matrix, calculates the detector measurements of I, Q, and U maps using the sky_decomp, takes the 2D FFT and azimuthally averages to power spectra. This function does not deconvolve the delta function bias that is present in get_leakage_beams. This function then plots the 1D spectra along with the instrument beam provided by the input parameters with a user defined normalization chosen by choose_normalization.
    
    returns plots and leakage spectra which are convolved with E and B maps (NOT solely the leakage beams!). returns an array of binned ells and a dictionary of spectra TT, EE, BB, TE, TB, and EB
    
    Inputs:
    ****************************************************************************
    convolved_coupled_beams [dictionary]: 3x3 beam coupling matrix
    pixel_size [float]: size of each pixel in arcmin
    N [int]: number of pixels along a map edge
    beam_fwhm [float]: 3dB point of gaussian beam
    sky_decomp [1D array]: a list of either the linear decomposition of the sky signal in IQU space or CMB realizations of I, Q, and U maps without instrument and observation effects
    delta_ell [int]: bin size for calculating power spectra
    ell_max [int]: maximum ell for power spectra calculation
    choose_normalization [str or 0]: if string it will be a key of the autospectra (TT for example), this will normalize every leakage spectra to the peak of this particular autospectra. If 0 is entered, each spectra is peak normalized.
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    binned_ell [1D array]: array of ell bins for averaging map powers
    binned_spectra_dict [dictionary]: dictionary of TT, TE, EE, etc spectra
    ****************************************************************************
    '''  
    #make instrument beam
   # inst_beam_1 = offset_2d_gaussian_beam(N, pixel_size, beam_fwhm,0,0)
    
    #Generate 1D power spectra from beam maps
    pix_size = pixel_size * 60. #pixel size in arcmin
#    Imap = sky_decomp[0] * convolved_coupled_beams['II'] + sky_decomp[1] * convolved_coupled_beams['IQ'] + sky_decomp[2] * convolved_coupled_beams['IU']
    Imap = sky_decomp[0] * convolved_coupled_beams['II'] + sky_decomp[1] * convolved_coupled_beams['IQ'] + sky_decomp[2] * convolved_coupled_beams['IU']
#    Qmap = sky_decomp[0] * convolved_coupled_beams['QI'] + sky_decomp[1] * convolved_coupled_beams['QQ'] + sky_decomp[2] * convolved_coupled_beams['QU']
    Qmap = sky_decomp[0] * convolved_coupled_beams['QI'] + sky_decomp[1] * convolved_coupled_beams['QQ'] + sky_decomp[2] * convolved_coupled_beams['UI']
#    Umap = sky_decomp[0] * convolved_coupled_beams['UI'] + sky_decomp[1] * convolved_coupled_beams['UQ'] + sky_decomp[2] * convolved_coupled_beams['UU']
    Umap = sky_decomp[0] * convolved_coupled_beams['QU'] + sky_decomp[1] * convolved_coupled_beams['UQ'] + sky_decomp[2] * convolved_coupled_beams['UU']
    
    #This now returns just 2d maps dictionary
    maps_dict = calculate_2d_spectra(Imap=Imap, Qmap=Qmap, Umap=Umap, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    binned_ell, binned_spectra_dict = bin_maps_to_1d(maps_dict, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
#    binned_ell, binned_spectra_dict = calculate_2d_spectra(Imap=Imap, Qmap=Qmap, Umap=Umap, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, unconvolved_beams=unconvolved_beams)

   # beam_maps_dict = calculate_2d_spectra(Imap=inst_beam_1, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    beam_maps_dict = calculate_2d_spectra(Imap=Imap, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    binned_ell, beam_spectrum = bin_maps_to_1d(beam_maps_dict, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)


    
    if choose_normalization == 0:
        #normalize all PS to 1
        binned_spectra_dict['TT'] = binned_spectra_dict['TT'][1:] / np.max(binned_spectra_dict['TT'][1:])
        binned_spectra_dict['EE'] = binned_spectra_dict['EE'][1:] / np.max(binned_spectra_dict['EE'][1:])
        binned_spectra_dict['BB'] = binned_spectra_dict['BB'][1:] / np.max(binned_spectra_dict['BB'][1:])
        binned_spectra_dict['TE'] = binned_spectra_dict['TE'][1:] / np.max(np.abs(binned_spectra_dict['TE'][1:]))
        binned_spectra_dict['EB'] = binned_spectra_dict['EB'][1:] / np.max(np.abs(binned_spectra_dict['EB'][1:]))
        binned_spectra_dict['TB'] = binned_spectra_dict['TB'][1:] / np.max(np.abs(binned_spectra_dict['TB'][1:]))
        beam_spectrum['TT'] = beam_spectrum['TT'][1:] / np.max(beam_spectrum['TT'][1:])

        if plots:
        #plot 
            plt.semilogy( binned_ell[1:], beam_spectrum['TT'] )
            plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] )
            plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] )
            plt.semilogy(binned_ell[1:], binned_spectra_dict['BB'])
            auto_labels = ['Beam','TT','EE','BB']
            plt.legend(auto_labels)
            plt.title('Auto Spectra and Instrument Beam')
            plt.ylabel('Window Function')
            plt.xlabel('$\ell$')
            #plt.ylim(3e-1,2)
            plt.xlim(0,ell_max)#5000)
            plt.show()

            zero_line = np.zeros(len(binned_spectra_dict['TE']))
            plt.plot( binned_ell[1:], binned_spectra_dict['TE'] )
            plt.plot(binned_ell[1:], binned_spectra_dict['EB'])
            plt.plot(binned_ell[1:], binned_spectra_dict['TB'])
            plt.plot(binned_ell[1:], zero_line, color='gray')
            cross_labels = ['T->E','E->B','T->B']
            plt.legend(cross_labels)
            plt.title('Leakage Spectra')
            plt.ylabel('Window Function')
            plt.xlabel('$\ell$')
            #plt.ylim(-1,1)
            plt.show()

            #Auto Power spectra as a fraction of the Instrument beam
            plt.semilogy( binned_ell[1:], beam_spectrum['TT'] / beam_spectrum['TT'] )
            plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] / beam_spectrum['TT'])
            plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] / beam_spectrum['TT'])
            plt.semilogy( binned_ell[1:], binned_spectra_dict['BB'] / beam_spectrum['TT'])
            legend_labels = ['Beam','TT','EE','BB']
            plt.legend(legend_labels)
            plt.ylabel('Beam Window Function Fraction')
            plt.xlabel('$\ell$')
            plt.title('Fraction of Beam')
            #plt.ylim(7e-1,1.3)
            plt.xlim(0,ell_max)#5000)
            plt.show()

    else:
        #normalize to leakage study
        if choose_normalization == 'TT':
            norm_fac = np.max(binned_spectra_dict['TT'][1:])
        elif choose_normalization == 'EE':
            norm_fac = np.max(binned_spectra_dict['EE'][1:])
        elif choose_normalization == 'BB':
            norm_fac = np.max(binned_spectra_dict['BB'][1:])
        else:
            print('Please use either TT, EE, or BB keys to study leakage effects')

        beam_spectrum['TT'] = beam_spectrum['TT'][1:] / np.max(beam_spectrum['TT'][1:])
        binned_spectra_dict['TT'] = binned_spectra_dict['TT'][1:] / norm_fac
        binned_spectra_dict['EE'] = binned_spectra_dict['EE'][1:] / norm_fac
        binned_spectra_dict['BB'] = binned_spectra_dict['BB'][1:] / norm_fac
        binned_spectra_dict['TE'] = binned_spectra_dict['TE'][1:] / (norm_fac*binned_spectra_dict['TT'])
        binned_spectra_dict['EB'] = binned_spectra_dict['EB'][1:] / (norm_fac*binned_spectra_dict['TT'])
        binned_spectra_dict['TB'] = binned_spectra_dict['TB'][1:] / (norm_fac*binned_spectra_dict['TT'])


        #plot
        if plots:
            plt.semilogy( binned_ell[1:], beam_spectrum['TT'] )
            plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] )
            plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] )
            plt.semilogy(binned_ell[1:], binned_spectra_dict['BB'])
            auto_labels = ['Beam','TT','EE','BB']
            plt.legend(auto_labels)
            plt.title('Auto Spectra and Instrument Beam')
            plt.ylabel('Window Function')
            plt.xlabel('$\ell$')
            #plt.ylim(1e-1,1.3)
            plt.xlim(0,ell_max)#5000)
            plt.show()

            zero_line = np.zeros(len(binned_spectra_dict['TE']))
            plt.plot( binned_ell[1:], binned_spectra_dict['TE'] )
            plt.plot(binned_ell[1:], binned_spectra_dict['EB'])
            plt.plot(binned_ell[1:], binned_spectra_dict['TB'])
            plt.plot(binned_ell[1:], zero_line, color='gray')
            cross_labels = ['TT->TE','EE->BB','TT->TB']
            plt.legend(cross_labels)
            plt.title('Leakage Spectra')
            plt.ylabel('Window Function')
            plt.xlabel('$\ell$')
            #plt.ylim(-1e-3,1e-3)
            plt.show()

            #Auto Power spectra as a fraction of the Instrument beam
            plt.semilogy( binned_ell[1:], beam_spectrum['TT'] / beam_spectrum['TT'] )
            plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] / beam_spectrum['TT'])
            plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] / beam_spectrum['TT'])
            plt.semilogy( binned_ell[1:], binned_spectra_dict['BB'] / beam_spectrum['TT'])
            legend_labels = ['Beam','TT','EE','BB']
            plt.legend(legend_labels)
            plt.ylabel('Beam Window Function Fraction')
            plt.xlabel('$\ell$')
            plt.title('Fraction of Beam')
            #plt.ylim(7e-1,1.3)
            plt.xlim(0,ell_max)#5000)
            plt.show()


    #unconvolved_beams = beam_matrix
    #return beam maps and raw power spectra
    return binned_ell[1:], binned_spectra_dict

def get_leakage_spectra2(convolved_coupled_beams, pixel_size, N, beam_fwhm, sky_decomp, delta_ell, ell_max, choose_normalization):
    
    #Cesiley's tweaks: made sure during normalization step, everything was just divided by the normalization factor once
     #              : removed plotting
        #           : added instrument beam TT to returns 
    '''
    from a given beam coupling matrix, calculates the detector measurements of I, Q, and U maps using the sky_decomp, takes the 2D FFT and azimuthally averages to power spectra. This function does not deconvolve the delta function bias that is present in get_leakage_beams. This function then plots the 1D spectra along with the instrument beam provided by the input parameters with a user defined normalization chosen by choose_normalization.
    
    returns plots and leakage spectra which are convolved with E and B maps (NOT solely the leakage beams!). returns an array of binned ells and a dictionary of spectra TT, EE, BB, TE, TB, and EB
    
    Inputs:
    ****************************************************************************
    convolved_coupled_beams [dictionary]: 3x3 beam coupling matrix
    pixel_size [float]: size of each pixel in arcmin
    N [int]: number of pixels along a map edge
    beam_fwhm [float]: 3dB point of gaussian beam
    sky_decomp [1D array]: a list of either the linear decomposition of the sky signal in IQU space or CMB realizations of I, Q, and U maps without instrument and observation effects
    delta_ell [int]: bin size for calculating power spectra
    ell_max [int]: maximum ell for power spectra calculation
    choose_normalization [str or 0]: if string it will be a key of the autospectra (TT for example), this will normalize every leakage spectra to the peak of this particular autospectra. If 0 is entered, each spectra is peak normalized.
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    binned_ell [1D array]: array of ell bins for averaging map powers
    binned_spectra_dict [dictionary]: dictionary of TT, TE, EE, etc spectra
    ****************************************************************************
    '''
    #make instrument beam
    inst_beam_1 = offset_2d_gaussian_beam(N, pixel_size, beam_fwhm,0,0)
    
    #Generate 1D power spectra from beam maps
    pix_size = pixel_size * 60. #pixel size in arcmin
    Imap = sky_decomp[0] * convolved_coupled_beams['II'] + sky_decomp[1] * convolved_coupled_beams['IQ'] + sky_decomp[2] * convolved_coupled_beams['IU']
    Qmap = sky_decomp[0] * convolved_coupled_beams['QI'] + sky_decomp[1] * convolved_coupled_beams['QQ'] + sky_decomp[2] * convolved_coupled_beams['QU']
    Umap = sky_decomp[0] * convolved_coupled_beams['UI'] + sky_decomp[1] * convolved_coupled_beams['UQ'] + sky_decomp[2] * convolved_coupled_beams['UU']
    
    #This now returns just 2d maps dictionary
    maps_dict = calculate_2d_spectra(Imap=Imap, Qmap=Qmap, Umap=Umap, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    
    binned_ell, binned_spectra_dict = bin_maps_to_1d(maps_dict, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
#    binned_ell, binned_spectra_dict = calculate_2d_spectra(Imap=Imap, Qmap=Qmap, Umap=Umap, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, unconvolved_beams=unconvolved_beams)

    beam_maps_dict = calculate_2d_spectra(Imap=inst_beam_1, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    binned_ell, beam_spectrum = bin_maps_to_1d(beam_maps_dict, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)


    
    if choose_normalization == 0:
        #normalize all PS to 1
        binned_spectra_dict['TT'] = binned_spectra_dict['TT'][1:] / np.nanmax(binned_spectra_dict['TT'][1:])
        binned_spectra_dict['EE'] = binned_spectra_dict['EE'][1:] / np.nanmax(binned_spectra_dict['EE'][1:])
        binned_spectra_dict['BB'] = binned_spectra_dict['BB'][1:] / np.nanmax(binned_spectra_dict['BB'][1:])
        binned_spectra_dict['TE'] = binned_spectra_dict['TE'][1:] / np.nanmax(np.abs(binned_spectra_dict['TE'][1:]))
        binned_spectra_dict['EB'] = binned_spectra_dict['EB'][1:] / np.nanmax(np.abs(binned_spectra_dict['EB'][1:]))
        binned_spectra_dict['TB'] = binned_spectra_dict['TB'][1:] / np.nanmax(np.abs(binned_spectra_dict['TB'][1:]))
        beam_spectrum['TT'] = beam_spectrum['TT'][1:] / np.nanmax(beam_spectrum['TT'][1:])

    else:
        #normalize to leakage study
        if choose_normalization == 'TT':
            norm_fac = binned_spectra_dict['TT'][1:]
        elif choose_normalization == 'EE':
            norm_fac = binned_spectra_dict['EE'][1:]
        elif choose_normalization == 'BB':
            norm_fac = binned_spectra_dict['BB'][1:]
        else:
            print('Please use either TT, EE, or BB keys to study leakage effects')

        beam_spectrum['TT'] = beam_spectrum['TT'][1:] / np.nanmax(beam_spectrum['TT'][1:])
        binned_spectra_dict['TT'] = binned_spectra_dict['TT'][1:] / norm_fac
        binned_spectra_dict['EE'] = binned_spectra_dict['EE'][1:] / norm_fac
        binned_spectra_dict['BB'] = binned_spectra_dict['BB'][1:] / norm_fac
        binned_spectra_dict['TE'] = binned_spectra_dict['TE'][1:] / (norm_fac)
        binned_spectra_dict['EB'] = binned_spectra_dict['EB'][1:] / (norm_fac)
        binned_spectra_dict['TB'] = binned_spectra_dict['TB'][1:] / (norm_fac)


    #unconvolved_beams = beam_matrix
    #return beam maps and raw power spectra
    return binned_ell[1:], binned_spectra_dict, beam_spectrum['TT']

def get_leakage_beams(convolved_coupled_beams, beam_matrix, pixel_size, N, beam_fwhm, sky_decomp, delta_ell, ell_max, choose_normalization):
    '''
    takes the 3x3 beam matrix (convolved_coupled_beams in this case) and generates the detector realization in IQU space using sky_decomp. Then calculates the delta function bias of the polarization components using beam matrix, which is the unconvolved form of convolved_coupled_beams that does not include the instrument beam. These pixel maps are used to deconvolve the delta function bias from the E and B maps. From here the T, E, and B maps are calculated from the detector maps and binned to 1D spectra. These spectra are then normalized in a particular manner by the user's choosing. 
    
    returns TT, EE, BB beams and leakage TE, TB, and EB beams
    
    Inputs:
    ****************************************************************************
    convolved_coupled_beams [dictionary]: 
    beam_matrix [dictionary]: 
    pixel_size [float]: size of each pixel in arcmin
    N [int]: number of pixels along edge of map
    beam_fwhm [float]: 3dB point width of gaussian beam
    sky_decomp [1D array]: either list of floats detailing magnitude of IQU stokes components in IQU space or a collection of CMB realized I, Q, and U maps
    delta_ell [int]: size of each spectral bin in ell
    ell_max [int]: maximum ell to go out to in binning procedure
    choose_normalization [string or 0]: if string it will be a key of the autospectra (TT for example), this will normalize every leakage spectra to the peak of this particular autospectra. If 0 is entered, each spectra is peak normalized.
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    
    ****************************************************************************
    '''
    
    #make instrument beam
    inst_beam_1 = offset_2d_gaussian_beam(N, pixel_size, beam_fwhm,0,0)
    
    #Generate 1D power spectra from beam maps
    pix_size = pixel_size * 60. #pixel size in arcmin
    Imap = sky_decomp[0] * convolved_coupled_beams['II'] + sky_decomp[1] * convolved_coupled_beams['IQ'] + sky_decomp[2] * convolved_coupled_beams['IU']
    Qmap = sky_decomp[0] * convolved_coupled_beams['QI'] + sky_decomp[1] * convolved_coupled_beams['QQ'] + sky_decomp[2] * convolved_coupled_beams['QU']
    Umap = sky_decomp[0] * convolved_coupled_beams['UI'] + sky_decomp[1] * convolved_coupled_beams['UQ'] + sky_decomp[2] * convolved_coupled_beams['UU']
    
    Qmap_deproj = sky_decomp[0] * beam_matrix['QI'] + sky_decomp[1] * beam_matrix['QQ'] + sky_decomp[2] * beam_matrix['QU']
    Umap_deproj = sky_decomp[0] * beam_matrix['UI'] + sky_decomp[1] * beam_matrix['UQ'] + sky_decomp[2] * beam_matrix['UU']
    
    #This now returns just 2d maps dictionary
    maps_dict = calculate_2d_leakage_beams(Imap=Imap, Qmap=Qmap, Umap=Umap, Qmap_deproj=Qmap_deproj, Umap_deproj=Umap_deproj, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    binned_ell, binned_spectra_dict = bin_maps_to_1d(maps_dict, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
#    binned_ell, binned_spectra_dict = calculate_2d_spectra(Imap=Imap, Qmap=Qmap, Umap=Umap, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, unconvolved_beams=unconvolved_beams)

    #no need to deproject from instrument beam
    beam_maps_dict = calculate_2d_spectra(Imap=inst_beam_1, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    binned_ell, beam_spectrum = bin_maps_to_1d(beam_maps_dict, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)

    if choose_normalization == 0:
        #normalize all PS to 1
        binned_spectra_dict['TT'] = binned_spectra_dict['TT'][1:] / np.max(binned_spectra_dict['TT'][1:])
        binned_spectra_dict['EE'] = binned_spectra_dict['EE'][1:] / np.max(binned_spectra_dict['EE'][1:])
        binned_spectra_dict['BB'] = binned_spectra_dict['BB'][1:] / np.max(binned_spectra_dict['BB'][1:])
        binned_spectra_dict['TE'] = binned_spectra_dict['TE'][1:] / np.max(np.abs(binned_spectra_dict['TE'][1:]))
        binned_spectra_dict['EB'] = binned_spectra_dict['EB'][1:] / np.max(np.abs(binned_spectra_dict['EB'][1:]))
        binned_spectra_dict['TB'] = binned_spectra_dict['TB'][1:] / np.max(np.abs(binned_spectra_dict['TB'][1:]))
        beam_spectrum['TT'] = beam_spectrum['TT'][1:] / np.max(beam_spectrum['TT'][1:])

        #plot 
        plt.semilogy( binned_ell[1:], beam_spectrum['TT'] )
        plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] )
        plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] )
        plt.semilogy(binned_ell[1:], binned_spectra_dict['BB'])
        auto_labels = ['Beam','TT','EE','BB']
        plt.legend(auto_labels)
        plt.title('Auto Spectra and Instrument Beam')
        plt.ylabel('Window Function')
        plt.xlabel('$\ell$')
        plt.ylim(3e-1,2)
        plt.xlim(0,5000)
        plt.show()

        zero_line = np.zeros(len(binned_spectra_dict['TE']))
        plt.plot( binned_ell[1:], binned_spectra_dict['TE'] )
        plt.plot(binned_ell[1:], binned_spectra_dict['EB'])
        plt.plot(binned_ell[1:], binned_spectra_dict['TB'])
        plt.plot(binned_ell[1:], zero_line, color='gray')
        cross_labels = ['T->E','E->B','T->B']
        plt.legend(cross_labels)
        plt.title('Leakage Spectra')
        plt.ylabel('Window Function')
        plt.xlabel('$\ell$')
        plt.ylim(-1,1)
        plt.show()

        #Auto Power spectra as a fraction of the Instrument beam
        plt.semilogy( binned_ell[1:], beam_spectrum['TT'] / beam_spectrum['TT'] )
        plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] / beam_spectrum['TT'])
        plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] / beam_spectrum['TT'])
        plt.semilogy( binned_ell[1:], binned_spectra_dict['BB'] / beam_spectrum['TT'])
        legend_labels = ['Beam','TT','EE','BB']
        plt.legend(legend_labels)
        plt.ylabel('Beam Window Function Fraction')
        plt.xlabel('$\ell$')
        plt.title('Fraction of Beam')
        plt.ylim(7e-1,1.3)
        plt.xlim(0,5000)
        plt.show()

    else:
        #normalize to leakage study
        if choose_normalization == 'TT':
            norm_fac = np.max(binned_spectra_dict['TT'][1:])
        elif choose_normalization == 'EE':
            norm_fac = np.max(binned_spectra_dict['EE'][1:])
        elif choose_normalization == 'BB':
            norm_fac = np.max(binned_spectra_dict['BB'][1:])
        else:
            print('Please use either TT, EE, or BB keys to study leakage effects')

        beam_spectrum['TT'] = beam_spectrum['TT'][1:] / np.max(beam_spectrum['TT'][1:])
        binned_spectra_dict['TT'] = binned_spectra_dict['TT'][1:] / norm_fac
        binned_spectra_dict['EE'] = binned_spectra_dict['EE'][1:] / norm_fac
        binned_spectra_dict['BB'] = binned_spectra_dict['BB'][1:] / norm_fac
        binned_spectra_dict['TE'] = binned_spectra_dict['TE'][1:] / (norm_fac*binned_spectra_dict['TT'])
        binned_spectra_dict['EB'] = binned_spectra_dict['EB'][1:] / (norm_fac*binned_spectra_dict['TT'])
        binned_spectra_dict['TB'] = binned_spectra_dict['TB'][1:] / (norm_fac*binned_spectra_dict['TT'])


        #plot
        plt.semilogy( binned_ell[1:], beam_spectrum['TT'] )
        plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] )
        plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] )
        plt.semilogy(binned_ell[1:], binned_spectra_dict['BB'])
        auto_labels = ['Beam','TT','EE','BB']
        plt.legend(auto_labels)
        plt.title('Auto Spectra and Instrument Beam')
        plt.ylabel('Window Function')
        plt.xlabel('$\ell$')
        plt.ylim(1e-1,1.3)
        plt.xlim(0,5000)
        plt.show()

        zero_line = np.zeros(len(binned_spectra_dict['TE']))
        plt.plot( binned_ell[1:], binned_spectra_dict['TE'] )
        plt.plot(binned_ell[1:], binned_spectra_dict['EB'])
        plt.plot(binned_ell[1:], binned_spectra_dict['TB'])
        plt.plot(binned_ell[1:], zero_line, color='gray')
        cross_labels = ['TT->TE','EE->BB','TT->TB']
        plt.legend(cross_labels)
        plt.title('Leakage Spectra')
        plt.ylabel('Window Function')
        plt.xlabel('$\ell$')
        plt.ylim(-1e-3,1e-3)
        plt.show()

        #Auto Power spectra as a fraction of the Instrument beam
        plt.semilogy( binned_ell[1:], beam_spectrum['TT'] / beam_spectrum['TT'] )
        plt.semilogy( binned_ell[1:], binned_spectra_dict['TT'] / beam_spectrum['TT'])
        plt.semilogy( binned_ell[1:], binned_spectra_dict['EE'] / beam_spectrum['TT'])
        plt.semilogy( binned_ell[1:], binned_spectra_dict['BB'] / beam_spectrum['TT'])
        legend_labels = ['Beam','TT','EE','BB']
        plt.legend(legend_labels)
        plt.ylabel('Beam Window Function Fraction')
        plt.xlabel('$\ell$')
        plt.title('Fraction of Beam')
        plt.ylim(7e-1,1.3)
        plt.xlim(0,5000)
        plt.show()
    
    

    
    return binned_ell, binned_spectra_dict

def convolve_pixel_with_beams(beam_matrix, N, pixel_size, beam_fwhm):
    
    '''
    convolves each component of the beam coupling matrix with a gaussian instrument beam and plots
    
    returns NA
    
    Inputs:
    ****************************************************************************
    beam_matrix [dictionary]: beam coupling matrix in IQU space
    N [int]: number of pixels along each side of map
    pixel_size [float]: size of each pixel in arcmin
    beam_fwhm [float]: 3dB point of gaussian instrument beam
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    NA
    ****************************************************************************
    '''
    
    #convolve with instrument beam and organize into dictionaries
    coupled_beams = {}
    inst_beam_1 = offset_2d_gaussian_beam(N, pixel_size, beam_fwhm,0,0)
    coupled_beams_keys = beam_matrix.keys()
    beams_list = [II_total, QI_total, UI_total, IQ_total, QQ_total, UQ_total, IU_total, QU_total, UU_total]
    convolved_coupled_beams = {}
    for key in coupled_beams_keys:
        convolved_coupled_beams[key] = convolve_pixel_instrument(beam_matrix[key], inst_beam_1)

    #Plot beam maps in 3x3 matrix
    fig, ax = plt.subplots(3,3, figsize=(20,20))
    
    #tiny offset to avoid numerical errors in log space
    eta = 0.0000001
    
    #make colormap relative to II beam
    vmin = 10.*np.log(np.min(II_total) + eta)
    vmax = 10.*np.log(np.max(II_total) + eta)

    #II
    ax[0,0].imshow(10. * np.log(np.abs(convolved_coupled_beams['II']) + eta),vmin=vmin,vmax=vmax)
    ax[0,0].set_title('II Beam')
    
    #IQ
    ax[0,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['IQ']) + eta),vmin=vmin,vmax=vmax)
    ax[0,1].set_title('IQ Beam')

    #IU
    ax[0,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['IU']) + eta),vmin=vmin,vmax=vmax)
    ax[0,2].set_title('IU beam')

    #QI
    ax[1,0].imshow(10.*np.log(np.abs(convolved_coupled_beams['QI']) + eta),vmin=vmin,vmax=vmax)
    ax[1,0].set_title('QI beam')

    #QQ
    ax[1,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['QQ']) + eta),vmin=vmin,vmax=vmax)
    ax[1,1].set_title('QQ beam')

    #QU
    ax[1,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['QU']) + eta),vmin=vmin,vmax=vmax)
    ax[1,2].set_title('QU beam')

    #UI
    ax[2,0].imshow(10. * np.log(np.abs(convolved_coupled_beams['UI']) + eta),vmin=vmin,vmax=vmax)
    ax[2,0].set_title('UI beam')

    #UQ
    ax[2,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['UQ']) + eta),vmin=vmin,vmax=vmax)
    ax[2,1].set_title('UQ beam')

    #UU
    ax[2,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['UU']) + eta),vmin=vmin,vmax=vmax)
    ax[2,2].set_title('UU beam')
    plt.savefig("Beam Matrix.png")
    plt.show()
    
    return


def plot_convolved_beams(det_dict, coupling_dict, freq1, freq2, pixel_size, perc_corr, N, beam_fwhm, TtoP_suppress):
    
    '''
    calculates the beams and then plots the 3x3 beam coupling matrix in real spcae on a log scale
    
    returns NA
    
    Inputs:
    ****************************************************************************
    det_dict [dictionary]: dictionary containing identifying information for the detectors in the focal plane
    coupling_dict [dictionary]: crosstalk matrix defining detector to detector coupling
    freq1 [float]: first band frequency 
    freq2 [float]: second band frequency
    pixel_size [float]: size of each pixel in arcmin
    perc_corr [float]: percent to which the detectors are coupling to each other
    N [int]: number of pixels along side of maps
    beam_fwhm [float]: the 3dB point of instrument beams
    TtoP_suppress [boolean]: sets the central value of T->P coupling maps to zero in where it otherwise may not be due to an imbalance of pair-differenced detectors
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    NA
    ****************************************************************************
    '''
    
    
    #calculate the beam matrix given a crosstalk matrix
    beam_matrix = calculate_beam_matrix(det_dict, coupling_dict, freq1, freq2, pixel_size, perc_corr, N, TtoP_suppress = TtoP_suppress)
    II_total = beam_matrix['II']
    IQ_total = beam_matrix['IQ']
    IU_total = beam_matrix['IU']
    QI_total = beam_matrix['QI']
    QQ_total = beam_matrix['QQ']
    QU_total = beam_matrix['QU']
    UI_total = beam_matrix['UI']
    UQ_total = beam_matrix['UQ']
    UU_total = beam_matrix['UU']

    #central values of beam maps
    print('central beam values II:' + str(II_total[int(N/2),int(N/2)]) + ', QQ:' + str(QQ_total[int(N/2),int(N/2)]) + ', UU:' + str(UU_total[int(N/2),int(N/2)]))
    print('IQ: ' + str(IQ_total[int(N/2),int(N/2)]))
    print('IU: ' + str(IU_total[int(N/2),int(N/2)]))
    print('QI: ' + str(QI_total[int(N/2),int(N/2)]))
    print('QU: ' + str(QU_total[int(N/2),int(N/2)]))
    print('UI: ' + str(UI_total[int(N/2),int(N/2)]))
    print('UQ: ' + str(UQ_total[int(N/2),int(N/2)]))

    #convolve with instrument beam and organize into dictionaries
    coupled_beams = {}
    inst_beam_1 = offset_2d_gaussian_beam(N, pixel_size, beam_fwhm,0,0)
    coupled_beams_keys = ['II','QI','UI','IQ','QQ','UQ','IU','QU','UU']
    beams_list = [II_total, QI_total, UI_total, IQ_total, QQ_total, UQ_total, IU_total, QU_total, UU_total]
    convolved_coupled_beams = {}
    for i in range(len(coupled_beams_keys)):
        coupled_beams[coupled_beams_keys[i]] = beams_list[i]

    for i in range(len(coupled_beams_keys)):
        convolved_coupled_beams[coupled_beams_keys[i]] = convolve_pixel_instrument(coupled_beams[coupled_beams_keys[i]],inst_beam_1)

    #Plot beam maps in 3x3 matrix
    fig, ax = plt.subplots(3,3, figsize=(20,20))
    
    #tiny offset to avoid numerical errors in log space
    eta = 0.0000001
    
    #make colormap relative to II beam
    vmin = 10.*np.log(np.min(II_total) + eta)
    vmax = 10.*np.log(np.max(II_total) + eta)

    #II
    ax[0,0].imshow(10. * np.log(np.abs(convolved_coupled_beams['II']) + eta),vmin=vmin,vmax=vmax)
    ax[0,0].set_title('II Beam')
    
    #IQ
    ax[0,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['IQ']) + eta),vmin=vmin,vmax=vmax)
    ax[0,1].set_title('IQ Beam')

    #IU
    ax[0,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['IU']) + eta),vmin=vmin,vmax=vmax)
    ax[0,2].set_title('IU beam')

    #QI
    ax[1,0].imshow(10.*np.log(np.abs(convolved_coupled_beams['QI']) + eta),vmin=vmin,vmax=vmax)
    ax[1,0].set_title('QI beam')

    #QQ
    ax[1,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['QQ']) + eta),vmin=vmin,vmax=vmax)
    ax[1,1].set_title('QQ beam')

    #QU
    ax[1,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['QU']) + eta),vmin=vmin,vmax=vmax)
    ax[1,2].set_title('QU beam')

    #UI
    ax[2,0].imshow(10. * np.log(np.abs(convolved_coupled_beams['UI']) + eta),vmin=vmin,vmax=vmax)
    ax[2,0].set_title('UI beam')

    #UQ
    ax[2,1].imshow(10. * np.log(np.abs(convolved_coupled_beams['UQ']) + eta),vmin=vmin,vmax=vmax)
    ax[2,1].set_title('UQ beam')

    #UU
    ax[2,2].imshow(10. * np.log(np.abs(convolved_coupled_beams['UU']) + eta),vmin=vmin,vmax=vmax)
    ax[2,2].set_title('UU beam')
    plt.savefig("Beam Matrix.png")
    plt.show()
    
    return


def calculate_beam_matrix(det_dict, coupling_dict, freq1, freq2, pixel_size, perc_corr, N, TtoP_suppress = False, to_fft=False):
    
    '''
    calculates the 3x3 beam coupling matrix in IQU space
    
    returns a dictionary of the beam coupling matrix
    
    Inputs:
    ****************************************************************************
    det_dict [dictionary]: dictionary containing identifying information for the detectors in the focal plane
    coupling_dict [dictionary]: crosstalk matrix that defines the detector to detector coupling
    freq1 [float]: first band frequency in consideration
    freq2 [float]: second band frequency in consideration
    pixel_size [float]: size of each pixel in map in arcmin
    perc_corr [float]: percent to which the detectors are coupled relative to the signal
    N [int]: number of pixels along edge of map
    TtoP_suppress [boolean]: if True the function sets the central values of the T->P coupling maps to zero where it may otherwise be nonzero due to an imbalance of pair-differenced orthogonal detectors on the focal plane. False does nothing
    to_fft [boolean]: *depcrecated does not do anything
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    beam_matrix [dictionary]: 3x3 beam coupling matrix in IQU space
    ****************************************************************************
    '''
   
    #number of detectors in focal plane
    num_det = len(det_dict[freq1].keys())
    
    num_det_x = 0
    num_det_y = 0
    num_det_a = 0
    num_det_b = 0
    for det in det_dict[freq1].keys():
        if det_dict[freq1][det]['ang'] == 0.:
            if det % 2. == 0.:
                num_det_x += 1
            else:
                num_det_y += 1
        if det_dict[freq1][det]['ang'] == np.pi/4.:
            if det % 2. == 0.:
                num_det_a += 1
            else:
                num_det_b += 1
    
    #initializations
    beam_map = np.zeros((N,N))
    II_total = beam_map.copy()
    IQ_total = beam_map.copy()
    IU_total = beam_map.copy()
    QI_total = beam_map.copy()
    QQ_total = beam_map.copy()
    QU_total = beam_map.copy()
    UI_total = beam_map.copy()
    UQ_total = beam_map.copy()
    UU_total = beam_map.copy()
    
    #for each frequency indices generate coupling matrix
    #freqs = [freq1, freq2]
    #coupling_dict = generate_random_coupling(det_dict)
    
#    coupling_dict = {}
#    for freq1 in freqs:
#        for freq2 in freqs:
#            coupling_dict[(freq1,freq2)] = {}
#            for det in det_dict[freq1].keys():
#                coupling_dict[(freq1,freq2)][det] = random.randint(0,len(det_dict[freq1].keys())-1)


    #loop thru every detector for x,y,a,b for each II,IQ,etc
    
    #pureI
    pixel_map_x_total = beam_map.copy()
    pixel_map_y_total = beam_map.copy()
    pixel_map_a_total = beam_map.copy()
    pixel_map_b_total = beam_map.copy()
    for det in det_dict[freq1].keys():

        pixel_map_x,pixel_map_y,pixel_map_a,pixel_map_b = crosstalk_pix_map(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        
        #lag detector maps
        #pixel_map_x = lag_map_along_scan(pixel_map_x, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y = lag_map_along_scan(pixel_map_y, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a = lag_map_along_scan(pixel_map_a, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b = lag_map_along_scan(pixel_map_b, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        ###################
        
        pixel_map_x_total += pixel_map_x
        pixel_map_y_total += pixel_map_y
        pixel_map_a_total += pixel_map_a
        pixel_map_b_total += pixel_map_b
    
    #average
    pixel_map_x_total = pixel_map_x_total / num_det
    pixel_map_y_total = pixel_map_y_total / num_det
    pixel_map_a_total = pixel_map_a_total / num_det
    pixel_map_b_total = pixel_map_b_total / num_det
    
    #first column
    norm_fac = np.max(pixel_map_x_total + pixel_map_y_total)
    II = (pixel_map_x_total + pixel_map_y_total) / norm_fac
    QI = (pixel_map_x_total - pixel_map_y_total) / norm_fac
    UI = (pixel_map_a_total - pixel_map_b_total) / norm_fac
    
    #choose whether output wants to include the T->P leakage that results from an imbalance of detectors aligned along axes
    if TtoP_suppress:
        QI[int(N/2.),int(N/2.)] = 0.
        UI[int(N/2.),int(N/2.)] = 0.
    
    time.sleep(0.5)
    
    #purQ
    pixel_map_x_total = beam_map.copy()
    pixel_map_y_total = beam_map.copy()
    pixel_map_a_total = beam_map.copy()
    pixel_map_b_total = beam_map.copy()
    for det in det_dict[freq1].keys():
    
        pixel_map_x1,pixel_map_y1,pixel_map_a1,pixel_map_b1 = crosstalk_pix_map_pureEx(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        pixel_map_x2,pixel_map_y2,pixel_map_a2,pixel_map_b2 = crosstalk_pix_map_pureEy(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        
        #lag detector maps
        #pixel_map_x1 = lag_map_along_scan(pixel_map_x1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y1 = lag_map_along_scan(pixel_map_y1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a1 = lag_map_along_scan(pixel_map_a1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b1 = lag_map_along_scan(pixel_map_b1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #
        #pixel_map_x2 = lag_map_along_scan(pixel_map_x2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y2 = lag_map_along_scan(pixel_map_y2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a2 = lag_map_along_scan(pixel_map_a2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b2 = lag_map_along_scan(pixel_map_b2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        ###################

        
        pixel_map_x_total += (pixel_map_x1)# + pixel_map_x2) / 2.
        pixel_map_y_total += (pixel_map_y1)# + pixel_map_y2) / 2.
        pixel_map_a_total += (pixel_map_a1)# + pixel_map_a2) / 2.
        pixel_map_b_total += (pixel_map_b1)# + pixel_map_b2) / 2.
    
    #average
    pixel_map_x_total = pixel_map_x_total / num_det
    pixel_map_y_total = pixel_map_y_total / num_det
    pixel_map_a_total = pixel_map_a_total / num_det
    pixel_map_b_total = pixel_map_b_total / num_det
    
    #second column
    #IQ_1 = (pixel_map_x_total + pixel_map_y_total) / (num_det_x * norm_fac / num_det)
    #QQ_1 = (pixel_map_x_total - pixel_map_y_total) / (num_det_x * norm_fac / num_det)
    #UQ_1 = (pixel_map_a_total - pixel_map_b_total) / (num_det_x * norm_fac / num_det)
    IQ_1 = (pixel_map_x_total + pixel_map_y_total) / (norm_fac)
    QQ_1 = (pixel_map_x_total - pixel_map_y_total) / (norm_fac)
    UQ_1 = (pixel_map_a_total - pixel_map_b_total) / (norm_fac)
    
    time.sleep(0.5)
    
    pixel_map_x_total = beam_map.copy()
    pixel_map_y_total = beam_map.copy()
    pixel_map_a_total = beam_map.copy()
    pixel_map_b_total = beam_map.copy()
    for det in det_dict[freq1].keys():
    
        pixel_map_x1,pixel_map_y1,pixel_map_a1,pixel_map_b1 = crosstalk_pix_map_pureEx(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        pixel_map_x2,pixel_map_y2,pixel_map_a2,pixel_map_b2 = crosstalk_pix_map_pureEy(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
       
        #lag detector maps
        #pixel_map_x1 = lag_map_along_scan(pixel_map_x1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y1 = lag_map_along_scan(pixel_map_y1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a1 = lag_map_along_scan(pixel_map_a1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b1 = lag_map_along_scan(pixel_map_b1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #
        #pixel_map_x2 = lag_map_along_scan(pixel_map_x2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y2 = lag_map_along_scan(pixel_map_y2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a2 = lag_map_along_scan(pixel_map_a2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b2 = lag_map_along_scan(pixel_map_b2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        ###################
    
        pixel_map_x_total += (pixel_map_x2)# + pixel_map_x2) / 2.
        pixel_map_y_total += (pixel_map_y2)# + pixel_map_y2) / 2.
        pixel_map_a_total += (pixel_map_a2)# + pixel_map_a2) / 2.
        pixel_map_b_total += (pixel_map_b2)# + pixel_map_b2) / 2.
    
    #average
    pixel_map_x_total = pixel_map_x_total / num_det
    pixel_map_y_total = pixel_map_y_total / num_det
    pixel_map_a_total = pixel_map_a_total / num_det
    pixel_map_b_total = pixel_map_b_total / num_det
    
    #second column
    #IQ_2 = (pixel_map_x_total + pixel_map_y_total) / (num_det_y * norm_fac / num_det)
    #QQ_2 = (pixel_map_x_total - pixel_map_y_total) / (num_det_y * norm_fac / num_det)
    #UQ_2 = (pixel_map_a_total - pixel_map_b_total) / (num_det_y * norm_fac / num_det)
    IQ_2 = (pixel_map_x_total + pixel_map_y_total) / (norm_fac)
    QQ_2 = (pixel_map_x_total - pixel_map_y_total) / (norm_fac)
    UQ_2 = (pixel_map_a_total - pixel_map_b_total) / (norm_fac)
    
    # divide by 2 is for average in the event there is an imbalance of detectors
    #IQ = (IQ_1 + IQ_2) / 2.
    #QQ = (np.abs(QQ_1) + np.abs(QQ_2)) / 2.
    #UQ = (UQ_1 + UQ_2) / 2.
    IQ = (IQ_1 + IQ_2)
    QQ = (np.abs(QQ_1) + np.abs(QQ_2))
    UQ = (UQ_1 + UQ_2)
    
    #Artificially Suppress Q->I coupling to avoid double counting
    IQ[int(N/2.),int(N/2.)] = 0.
    
    time.sleep(0.5)
    
    #pureU
    pixel_map_x_total = beam_map.copy()
    pixel_map_y_total = beam_map.copy()
    pixel_map_a_total = beam_map.copy()
    pixel_map_b_total = beam_map.copy()
    for det in det_dict[freq1].keys():
    
        pixel_map_x1,pixel_map_y1,pixel_map_a1,pixel_map_b1 = crosstalk_pix_map_pureEa(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        pixel_map_x2,pixel_map_y2,pixel_map_a2,pixel_map_b2 = crosstalk_pix_map_pureEb(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        
        #lag detector maps
        #pixel_map_x1 = lag_map_along_scan(pixel_map_x1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y1 = lag_map_along_scan(pixel_map_y1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a1 = lag_map_along_scan(pixel_map_a1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b1 = lag_map_along_scan(pixel_map_b1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #
        #pixel_map_x2 = lag_map_along_scan(pixel_map_x2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y2 = lag_map_along_scan(pixel_map_y2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a2 = lag_map_along_scan(pixel_map_a2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b2 = lag_map_along_scan(pixel_map_b2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        ###################
        
        pixel_map_x_total += (pixel_map_x1)# + pixel_map_x2) / 2.
        pixel_map_y_total += (pixel_map_y1)# + pixel_map_y2) / 2.
        pixel_map_a_total += (pixel_map_a1)# + pixel_map_a2) / 2.
        pixel_map_b_total += (pixel_map_b1)# + pixel_map_b2) / 2.

    #average
    pixel_map_x_total = pixel_map_x_total / num_det
    pixel_map_y_total = pixel_map_y_total / num_det
    pixel_map_a_total = pixel_map_a_total / num_det
    pixel_map_b_total = pixel_map_b_total / num_det
    
    #third column
    #IU_1 = (pixel_map_x_total + pixel_map_y_total) / (num_det_a * norm_fac / num_det)
    #QU_1 = (pixel_map_x_total - pixel_map_y_total) / (num_det_a * norm_fac / num_det)
    #UU_1 = (pixel_map_a_total - pixel_map_b_total) / (num_det_a * norm_fac / num_det)
    IU_1 = (pixel_map_x_total + pixel_map_y_total) / (norm_fac)
    QU_1 = (pixel_map_x_total - pixel_map_y_total) / (norm_fac)
    UU_1 = (pixel_map_a_total - pixel_map_b_total) / (norm_fac)
    
    time.sleep(0.5)
    
    pixel_map_x_total = beam_map.copy()
    pixel_map_y_total = beam_map.copy()
    pixel_map_a_total = beam_map.copy()
    pixel_map_b_total = beam_map.copy()
    for det in det_dict[freq1].keys():
    
        pixel_map_x1,pixel_map_y1,pixel_map_a1,pixel_map_b1 = crosstalk_pix_map_pureEa(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        pixel_map_x2,pixel_map_y2,pixel_map_a2,pixel_map_b2 = crosstalk_pix_map_pureEb(det, det_dict, coupling_dict, freq1, freq2, perc_corr, beam_map,pixel_size)
        
        #lag detector maps
        #pixel_map_x1 = lag_map_along_scan(pixel_map_x1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y1 = lag_map_along_scan(pixel_map_y1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a1 = lag_map_along_scan(pixel_map_a1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b1 = lag_map_along_scan(pixel_map_b1, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #
        #pixel_map_x2 = lag_map_along_scan(pixel_map_x2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_y2 = lag_map_along_scan(pixel_map_y2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_a2 = lag_map_along_scan(pixel_map_a2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        #pixel_map_b2 = lag_map_along_scan(pixel_map_b2, det_dict[freq1][det]['tau'], pix_size = pixel_size)
        ###################
        
        pixel_map_x_total += (pixel_map_x2)# + pixel_map_x2) / 2.
        pixel_map_y_total += (pixel_map_y2)# + pixel_map_y2) / 2.
        pixel_map_a_total += (pixel_map_a2)# + pixel_map_a2) / 2.
        pixel_map_b_total += (pixel_map_b2)# + pixel_map_b2) / 2.
    
    #average
    pixel_map_x_total = pixel_map_x_total / num_det
    pixel_map_y_total = pixel_map_y_total / num_det
    pixel_map_a_total = pixel_map_a_total / num_det
    pixel_map_b_total = pixel_map_b_total / num_det
    
    #third column
    #IU_2 = (pixel_map_x_total + pixel_map_y_total) / (num_det_b * norm_fac / num_det)
    #QU_2 = (pixel_map_x_total - pixel_map_y_total) / (num_det_b * norm_fac / num_det)
    #UU_2 = (pixel_map_a_total - pixel_map_b_total) / (num_det_b * norm_fac / num_det)
    IU_2 = (pixel_map_x_total + pixel_map_y_total) / (norm_fac)
    QU_2 = (pixel_map_x_total - pixel_map_y_total) / (norm_fac)
    UU_2 = (pixel_map_a_total - pixel_map_b_total) / (norm_fac)
    
    # divide by 2 is for average in the event there is an imbalance of detectors
    #IU = (IU_1 + IU_2) / 2.
    #QU = (QU_1 + QU_2) / 2.
    #UU = (np.abs(UU_1) + np.abs(UU_2)) / 2.
    IU = (IU_1 + IU_2)
    QU = (QU_1 + QU_2)
    UU = (np.abs(UU_1) + np.abs(UU_2))
    
    #Artificially suppress U->I coupling to avoid double counting
    IU[int(N/2.),int(N/2.)] = 0.
    
    beam_keys = ['II','IQ','IU','QI','QQ','QU','UI','UQ','UU']
    beams = [II,IQ,IU,QI,QQ,QU,UI,UQ,UU]
    
    beam_matrix = {}
    beam_matrix_fft = {}
    for i in range(len(beam_keys)):
        beam_matrix[beam_keys[i]] = beams[i]
    
    return beam_matrix

def return_IQU_fft(sky_decomposition, beam_matrix):
    '''
    returns tuple of fft of IQU pixel maps without the instrument beam for later deconvolution of delta function biases
    
    Inputs:
    ****************************************************************************
    sky_decomposition [1D array]: the decomposition of the sky signal in IQU space
    beam_matrix [dictionary]: the 3x3 beam coupling matrix in IQU space
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    I_fft [2D array]: The 2D FFT of the pixel beam I that the focal plane sees
    Q_fft [2D array]: The 2D FFT of the pixel beam Q that the focal plane sees
    U_fft [2D array]: The 2D FFT of the pixel beam U that the focal plane sees
    ****************************************************************************
    '''  
    I = sky_decomposition[0]*beam_matrix['II'] + sky_decomposition[1]*beam_matrix['IQ'] + sky_decomposition[2]*beam_matrix['IU']
                            
    Q = sky_decomposition[0]*beam_matrix['QI'] + sky_decomposition[1]*beam_matrix['QQ'] + sky_decomposition[2]*beam_matrix['QU']
                            
    U = sky_decomposition[0]*beam_matrix['UI'] + sky_decomposition[1]*beam_matrix['UQ'] + sky_decomposition[2]*beam_matrix['UU']

    I_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(I)))
    Q_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Q)))
    U_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U)))
    
    return (I_fft, Q_fft, U_fft)

def generate_random_coupling(det_dict):
    
    '''
    generates a crosstalk matrix detailing detector to detector coupling via a random mechanism. This means each detector is coupled to one other random detector in the focal plane
    
    returns the crosstalk matrix for random coupling
    
    Inputs:
    ****************************************************************************
    det_dict [dictionary]: dictionary containing identifying information on the detectors in the focal plane
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    coupling_dict [dictionary]: the crosstalk matrix defining the detector to detector coupling
    ****************************************************************************
    '''
    freqs = det_dict.keys()
    coupling_dict = {}
    for freq1 in freqs:
        for freq2 in freqs:
            coupling_dict[(freq1,freq2)] = {}
            for det in det_dict[freq1].keys():
                coupling_dict[(freq1,freq2)][det] = random.randint(0,len(det_dict[freq1].keys())-1)

    
    return coupling_dict

def generate_focal_plane_distribution(path_to_positions, num_det, freqs, rescale):
    
    '''
    generates the focal plane distribution given by the associated input text file
    
    returns a dictionary of identifying information for each detector including numerology, position, band frequency, polarization angle, and signal magnitude
    
    Inputs:
    ****************************************************************************
    path_to_positions [string]: the path and filename to find where the text file containing the positional information is located for the focal plane
    num_det [int]: the number of detectors
    freqs [1D array]: array of band frequencies
    rescale [float]: amount to globally scale the positions relative to the center of the focal plane
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    det_dict [dictionary]: dictionary of identifying information for detectors in focal plane
    ****************************************************************************
    '''
    
#    import random
    
    #Load spatial data

    #load from csv
    det_pos = pd.read_csv(path_to_positions, sep = ' ', names = ['ID','x','y','pol_ang'])
    deg_to_rad = np.pi / 180.

    x_pos_freq1 = []
    y_pos_freq1 = []
    x_pos_freq2 = []
    y_pos_freq2 = []
    det_ID_freq1 = []
    det_ID_freq2 = []
    det_pol_ang_freq1 = []
    det_pol_ang_freq2 = []
    for i in range(0,num_det,4):

        #90 GHz
        x_pos_freq1.append( rescale * det_pos['x'][i] )
        x_pos_freq1.append( rescale * det_pos['x'][i+1] )
        y_pos_freq1.append( rescale * det_pos['y'][i] )
        y_pos_freq1.append( rescale * det_pos['y'][i+1] )
        det_ID_freq1.append( det_pos['ID'][i] )
        det_ID_freq1.append( det_pos['ID'][i+1] )
        #round pol angle to nearest 0.x b/c of numerical errors
        det_pol_ang_freq1.append( round(det_pos['pol_ang'][i],1) )
        det_pol_ang_freq1.append( round(det_pos['pol_ang'][i+1],1) )


        #150 GHz
        x_pos_freq2.append( rescale * det_pos['x'][i+2] )
        x_pos_freq2.append( rescale * det_pos['x'][i+3] )
        y_pos_freq2.append( rescale * det_pos['y'][i+2] )
        y_pos_freq2.append( rescale * det_pos['y'][i+3] )
        det_ID_freq2.append( det_pos['ID'][i+2] )
        det_ID_freq2.append( det_pos['ID'][i+3] )
        #round pol angle to nearest 0.x b/c of numerical errors
        det_pol_ang_freq2.append( round(det_pos['pol_ang'][i+2],1) )
        det_pol_ang_freq2.append( round(det_pos['pol_ang'][i+3],1) )

    #label pol angle information into xyab axes in degrees
    #global_x_ang = np.min(det_pol_ang_freq1)
    #global_y_ang = global_x_ang + 90
    #global_a_ang = global_x_ang + 45
    #global_b_ang = global_a_ang + 90

    #num_det = len(det_pos['x']) #change to number of rows in spatial data
    det_list = np.arange(1,len(x_pos_freq1),1)

    det_dict = {}
    det_dict[freqs[0]] = {}
    det_dict[freqs[1]] = {}
    for freq in freqs:
        for det in range(len(x_pos_freq1)):

            det_dict[freqs[0]][det] = {}
            det_dict[freqs[0]][det]['x'] = x_pos_freq1[det]
            det_dict[freqs[0]][det]['y'] = y_pos_freq1[det]
            det_dict[freqs[0]][det]['ID'] = det_ID_freq1[det]
            #python uses radians in trig functions
            det_dict[freqs[0]][det]['ang'] = det_pol_ang_freq1[det]*deg_to_rad
            det_dict[freqs[0]][det]['sig'] = 1.
            det_dict[freqs[0]][det]['status'] = 'alive'
            #if det_dict[freqs[0]][det]['ang'] == global_x_ang:
            #    det_dict[freqs[0]][det]['axis'] = 'x'
            #elif det_dict[freqs[0]][det]['ang'] == global_y_ang:
            #    det_dict[freqs[0]][det]['axis'] = 'y'
            #elif det_dict[freqs[0]][det]['ang'] == global_a_ang:
            #    det_dict[freqs[0]][det]['axis'] = 'a'
            #elif det_dict[freqs[0]][det]['ang'] == global_b_ang:
            #    det_dict[freqs[0]][det]['axis'] = 'b'
            #else:
            #    print('no pol angle information found')

            det_dict[freqs[1]][det] = {}
            det_dict[freqs[1]][det]['x'] = x_pos_freq2[det]
            det_dict[freqs[1]][det]['y'] = y_pos_freq2[det]
            det_dict[freqs[1]][det]['ID'] = det_ID_freq2[det]
            #python uses radians for trig functions
            det_dict[freqs[1]][det]['ang'] = det_pol_ang_freq2[det]*deg_to_rad
            det_dict[freqs[1]][det]['sig'] = 1.
            det_dict[freqs[1]][det]['status'] = 'alive'
            #if det_dict[freqs[1]][det]['ang'] == global_x_ang:
            #    det_dict[freqs[1]][det]['axis'] = 'x'
            #elif det_dict[freqs[1]][det]['ang'] == global_y_ang:
            #    det_dict[freqs[1]][det]['axis'] = 'y'
            #elif det_dict[freqs[1]][det]['ang'] == global_a_ang:
            #    det_dict[freqs[1]][det]['axis'] = 'a'
            #elif det_dict[freqs[1]][det]['ang'] == global_b_ang:
            #    det_dict[freqs[1]][det]['axis'] = 'b'
            #else:
            #    print('no pol angle information found')

    print('The wafer information has a spread of ' + str(np.max(x_pos_freq1) - np.min(x_pos_freq1)) + ' degrees in x')
    print('The wafer information has a spread of ' + str(np.max(y_pos_freq1) - np.min(y_pos_freq1)) + ' degrees in y')
    #assign detector taus
    #define detector time constants (sampling from a gamma distribution)
    #det_dict = sample_det_tau(det_dict)
    
    return det_dict

def kill_pixels(det_IDs, det_dict):
    
    '''
    eliminates the signal for a list of detectors present in a focal plane
    
    returns the detector dictionary updated with the associated killed detectors
    
    Inputs:
    ****************************************************************************
    det_IDs [1D array]: array of detectors indexed by identifying number who's signal should be killed
    det_dict [dictionary]: dictionary of identifying information for each detector in a focal plane
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    det_dict [dictionary]: updated dictionary of identifying information for each detector in a focal plane
    ****************************************************************************
    '''
    
    for det_ID in det_IDs:
        for freq in det_dict.keys():
            for det in det_dict[freq].keys():
                if det_dict[freq][det]['ID'] == det_ID:
                    det_dict[freq][det]['sig'] = 0.
                    det_dict[freq][det]['status'] = 'dead'
    
    return det_dict

#for rhombus layout
def map_det_pos_to_bondpads(det_dict, num_det_per_rhom = 576, num_bond_sets_per_side=12):
    
    '''
    *deprecated do not use*
    '''
    
    return


def generate_bondpad_coupling_rhombus(det_dict):
    
    '''
    generates the crosstalk matrix for wafers fabricated in the rhombus layout
    
    returns the crosstalk matrix defining the detector to detector coupling for a rhombus layout wafer where detectors in adjacent SQUID readout columns are coupled
    
    Inputs:
    ****************************************************************************
    det_dict [dictionary]: the dictionary containing the identifying information for each detector
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    coupling_dict [dictionary]: the crosstalk matrix defining the detector to detector coupling
    ****************************************************************************
    '''
    #organize pixel indices into readout columns for rhombus layout
    #for rows for 90 GHz
    row_inds = {}

    for l in range(3):
        indexer_rhom = [0,144,288]
        row_inds[l] = {}
        for j in range(12):
            indexer_row = np.arange(j+2,j+14)
            start_index = [0,1,3,6,10,15,21,28,36,45,55,66,78]
            i=0
            row_list=[]
            for k in range(12):
                row_list.append( start_index[j] + i + indexer_rhom[l] )

                i += indexer_row[k]

            if j != 0:
                correction_list = [1,4,9,16,25,36,49,64,81,100,121]
                row_list[-j:] -= np.asarray(correction_list[:j])
            row_inds[l][j] = row_list


    #for columns for 150 GHz
    col_inds = {}

    for l in range(3):
        indexer_rhom = [0,144,288]
        col_inds[l] = {}
        for j in range(12):
            indexer_row = np.arange(j+1,j+13)
            start_index = [0,2,5,9,14,20,27,35,44,54,65,77]
            i=0
            col_list=[]
            for k in range(12):
                col_list.append( start_index[j] + i + indexer_rhom[l] )

                i += indexer_row[k]

            if j != 0:
                correction_list = [1,4,9,16,25,36,49,64,81,100,121]
                col_list[-j:] -= np.asarray(correction_list[:j])
            col_inds[l][j] = col_list
            
          
    #organize pixel indices into detector indices for readout columns
    freqs = list(det_dict.keys())
    #num_det_per_rhom = 576

    #first organize det_dict into 3 separate rhombuses
    #rhom_layout_dict[rhom#][freq][row/column]
    #this is done by grouping pairs of bondpads and assigning a detector number wired to it
    #BECAREFUL: Wiring may change so think about how to make modular?
    #FIXIT wiring should be opposite of numbering, i.e. the highest number pixels should be wired to the farthest left bondpads
    #think about writing dictionaries to toml files and interface with Sara on formatting
    rhom_layout_dict = {}
    for i in range(3):
        rhom_layout_dict[i] = {}
        for freq in freqs:
            det_num_list = list(det_dict[freq].keys())
            rhom_layout_dict[i][freq] = {}
            #store detector list every 24 rows
            #there are 12 rows per side or 24 per rhombus
            #FIXIT: I should figure out which is rows are indexed first
            #this maps each frequency to a different side of the rhombus
            for j in range(12):

                if freq == freqs[0]:
                    #select out x and y detector indices
                    x_det_inds = np.asarray(row_inds[i][j])*2
                    y_det_inds = np.asarray(row_inds[i][j])*2+1
                    det_inds_list = list(x_det_inds) + list(y_det_inds)
                    det_inds_list.sort()

                    rhom_layout_dict[i][freq][j] = det_inds_list


                elif freq == freqs[1]:
                    #select out x and y detector indices
                    x_det_inds = np.asarray(col_inds[i][j])*2
                    y_det_inds = np.asarray(col_inds[i][j])*2+1
                    det_inds_list = list(x_det_inds) + list(y_det_inds)
                    det_inds_list.sort()

                    rhom_layout_dict[i][freq][j] = det_inds_list
    
    
    #finally generate coupling dict
    #detectors within one column (12 set of 50 bondpads) are coupled to the next adjacent row (not cyclic and is unidirectional)
    #there are 5 columns of 60 detectors in readout modules
    i = 0
    coupling_dict = {}
    for rhombus in rhom_layout_dict.keys():
        for freq1 in rhom_layout_dict[rhombus].keys():
            for freq2 in rhom_layout_dict[rhombus].keys():
                coupling_dict[(freq1, freq2)] = {}
    for rhombus in rhom_layout_dict.keys():
        for freq in rhom_layout_dict[rhombus].keys():
            for row_num in rhom_layout_dict[rhombus][freq]:

                for j in range(len(rhom_layout_dict[rhombus][freq][row_num])):
                    det_num = rhom_layout_dict[rhombus][freq][row_num][j]
                    if i==64:
                        coupling_dict[(freq, freq)][det_num] = 'na'
                        i=0
                    else:
                        try:
                            coupling_dict[(freq, freq)][det_num] = rhom_layout_dict[rhombus][freq][row_num][j+1]
                        except (KeyError, IndexError):
                            if row_num == 11:
                                coupling_dict[(freq, freq)][det_num] = 'na'
                            else:
                                coupling_dict[(freq, freq)][det_num] = rhom_layout_dict[rhombus][freq][row_num+1][0]

                    i+=1

    return coupling_dict


def generate_random_focal_plane_distribution(path_to_positions, num_det, freqs, rescale):
    
    '''
    generates a dictionary for a focal plane with a random distribution of x-y and a-b polarization angle detectors
    
    returns a dictionary containing the identifying information for each detector including numerology, position, band frequency, polarization angle, and signal magnitude
    
    Inputs:
    ****************************************************************************
    path_to_positions [string]: the file path, including the filename, that contains the information of how the detectors are placed on the sky
    num_det [int]: number of detectors
    freqs [float]: list of band frequencies there is positional information for
    rescale [float]: factor to scale the position data on the sky
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    det_dict [dictionary]: dictionary of identifying information for each detector including numerology, position, band frequency, polarization angle, and signal magnitude
    ****************************************************************************
    '''
    #import random
    
    #Load spatial data

    #load from csv
    det_pos = pd.read_csv(path_to_positions, sep = ' ', names = ['x','y'])

    x_offsets_90 = []
    y_offsets_90 = []
    x_offsets_150 = []
    y_offsets_150 = []
    for i in range(0,num_det,4):

        #90 GHz
        x_offsets_90.append( rescale * det_pos['x'][i] )
        x_offsets_90.append( rescale * det_pos['x'][i+1] )
        y_offsets_90.append( rescale * det_pos['y'][i] )
        y_offsets_90.append( rescale * det_pos['y'][i+1] )


        #150 GHz
        x_offsets_150.append( rescale * det_pos['x'][i+2] )
        x_offsets_150.append( rescale * det_pos['x'][i+3] )
        y_offsets_150.append( rescale * det_pos['y'][i+2] )
        y_offsets_150.append( rescale * det_pos['y'][i+3] )

    #num_det = len(det_pos['x']) #change to number of rows in spatial data
    det_list = np.arange(1,len(x_offsets_90),1)

    det_dict = {}
    det_dict[freqs[0]] = {}
    det_dict[freqs[1]] = {}
    for freq in freqs:
        for det in range(len(x_offsets_90)):

            det_dict[freqs[0]][det] = {}
            det_dict[freqs[0]][det]['x'] = x_offsets_90[det]
            det_dict[freqs[0]][det]['y'] = y_offsets_90[det]

            det_dict[freqs[1]][det] = {}
            det_dict[freqs[1]][det]['x'] = x_offsets_150[det]
            det_dict[freqs[1]][det]['y'] = y_offsets_150[det]

        
    #assign signals and detector distribution layout on focal plane
    #generate random signal for each detector
    #also generate angles by which the local pixel antennae are distributed relative to each other
    #freqs = [freq1, freq2]
    for freq1 in freqs:
        for det in det_dict[freq1].keys():
            det_dict[freq1][det]['sig'] = 1.
            #det_dict[det]['sigy'] = 1.

            orientation = random.randint(1,2)

            if orientation == 1:
                det_dict[freq1][det]['ang'] = 0.
                #det_dict[det]['angy'] = 0.

            elif orientation == 2:
                det_dict[freq1][det]['ang'] = np.pi / 4.
                #det_dict[det]['angy'] = np.pi / 4.

    #assign detector taus
    #define detector time constants (sampling from a gamma distribution)
    det_dict = sample_det_tau(det_dict)

    return det_dict

 ###############################  ###############################  ############################### 

#############################################Beam Modules################################################
def convolve_pixel_instrument(beam_map, inst_beam_map):
    
    '''
    convolves the pixel beam map with the instrument beam
    
    returns the convolved map in real space
    
    Inputs:
    ****************************************************************************
    beam_map [2D array]: the pixel beam map without the gaussian instrument beam that includes systematics
    inst_beam_map [2D array]: the instrument beam generated for this analysis
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    convolved_map [2D array]: the map that convolves the instrument beam map with the pixel beam map
    ****************************************************************************
    '''

    #fft of beam_map
    beam_map_fft = np.fft.fft2(np.fft.fftshift(beam_map))
    
    #fft of inst_beam_map
    inst_beam_map_fft = np.fft.fft2(np.fft.fftshift(inst_beam_map))
    
    #convolve
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(beam_map_fft * inst_beam_map_fft)))
       
    return convolved_map
 

def make_2d_gaussian_beam(N,pix_size,beam_size_fwhp):
    
    '''
    generates a gaussian beam
    
    returns a 2D array gaussian beam map
    
    Inputs:
    ****************************************************************************
    N [int]: number of pixels along the edge of the map
    pix_size [float]: size of each pixel in arcmin
    beam_size_fwhp [float]: -3 dB point width of gaussian in arcmin
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    gaussian [2D array]: gaussian beam map
    ****************************************************************************
    '''
     # make a 2d coordinate system
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
    #plt.title('Radial co ordinates')
    #plt.imshow(R)
  
    # make a 2d gaussian 
    beam_sigma = beam_size_fwhp / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.)
    gaussian = gaussian / np.sum(gaussian)
    # return the gaussian
    #plt.imshow(gaussian)
    return(gaussian)
 
    
def offset_2d_gaussian_beam(N, pix_size, pixel_beam_fwhp, x_offset, y_offset):
    
    '''
    generates a 2D gaussian beam in real space and contains the ability to offset the center of the beam relative to the map center
    
    returns the offset gaussian beam
    
    Inputs:
    ****************************************************************************
    N [int]: number of pixels along the side of the map
    pix_size [float]: size of each pixel in arcmin
    pixel_beam_fwhp [float]: -3 dB point of gaussian beam in arcmin
    x_offset [float]: magnitude of offset along the x axis in arcmin
    y_offset [float]: magnitude of offset along the y axis in arcmin
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    pixel_beam [2D array]: offset gaussian beam map
    ****************************************************************************
    '''

    # make a 2d coordinate system
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X) + y_offset
    R = np.sqrt((X+x_offset)**2. + (Y+y_offset)**2.)
    #plt.title('Radial co ordinates')
    #plt.imshow(R)
    
    beam_sigma = pixel_beam_fwhp / np.sqrt(8.*np.log(2))
    pixel_beam = np.exp(-0.5 * (R/beam_sigma)**2.)
    #pixel_beam = pixel_beam / np.sum(pixel_beam)
    
    pixel_beam = pixel_beam / np.max(pixel_beam)
    
    return pixel_beam

 ###############################  ###############################  ############################### 
    
#############################################Crosstalk Modules################################################

def crosstalk_pix_map(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    
    '''*Deprecated, Do not use*'''

    #number of pixels on each map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    
    
    #crosstalk depends on relative orientation between detectors
    if coupling_dict[(freq1,freq2)][det] == 'na':
        crosstalk = 0
        
        i = 0
        j = 0
    else:
        #take coupled value and calculate crosstalk
        coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
        
        #add to the relative offset in this pixel map
        i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
        j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
    
    
  
    #make x, y, a, b maps
    det_ang = det_dict[freq1][det]['ang']
    
    #for contribution to global x,y,a,b
    pixel_map_x[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - 0.))**2.
    pixel_map_y[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - np.pi/2.))**2.
    pixel_map_a[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - np.pi/4.))**2.
    pixel_map_b[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - 3*np.pi/4.))**2.
    
    #add crosstalk signal projections
    pixel_map_x[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 0.))**2.
    pixel_map_y[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/2.))**2.
    pixel_map_a[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/4.))**2.
    pixel_map_b[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 3*np.pi/4.))**2.
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEx(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    
    '''
    calculates the crosstalk contribution to the global on sky (x) axis from a particular detector. the contribution is calculated by the trigonometry from the alignment of the detector, cross detector with the global x, y, a, b axes. 
    
    returns pixel maps with the new crosstalk contribution position and magnitude added
    
    Inputs:
    ****************************************************************************
    det [int]: the identifying number of the detector in question being crosstalked to
    det_dict [dictionary]: focal plane information storing detector number along with frequency response, angle relative to global sky coordinates, and signal magnitude
    coupling_dict [dictionary]: the crosstalk matrix that defines the detector to detector coupling
    freq1 [int]: first frequency being considered
    freq2 [int]: second frequency being considered, for crosstalk between adjacent bands these two numbers would be different
    perc_corr [float]: the percent to which the signals of the detectors are correlated
    empty_beam_map [2D array]: a copy of null beam maps to size the pixel beam maps appropriately
    pixel_size [float]: size of each pixel in the map in arcmin
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    pixel_map_x [2D array]: contribution of this detector and crosstalk signal along the global sky x axis
    pixel_map_y [2D array]: contribution of this detector and crosstalk signal along the global sky y axis
    pixel_map_a [2D array]: contribution of this detector and crosstalk signal along the global sky a axis
    pixel_map_b [2D array]: contribution of this detector and crosstalk signal along the global sky b axis
    ****************************************************************************
    '''
    
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    

    #crosstalk depends on relative orientation between detectors
    if coupling_dict[(freq1,freq2)][det] == 'na':
        crosstalk = 0
        
        i = 0
        j = 0
        
    else:
        #take coupled value and calculate crosstalk
        coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
    
        #add to the relative offset in this pixel map
        i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
        j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  
    det_ang = det_dict[freq1][det]['ang']
    
    #for contribution to global x,y,a,b for pure Ex signal
    pixel_map_x[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - 0.))**2.
    pixel_map_y[int(N/2),int(N/2)] = 0.
    pixel_map_a[int(N/2),int(N/2)] = 0.
    pixel_map_b[int(N/2),int(N/2)] = 0.
    
    #add crosstalk signal projections
    pixel_map_x[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 0.))**2.
    pixel_map_y[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/2.))**2.
    pixel_map_a[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/4.))**2.
    pixel_map_b[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 3*np.pi/4))**2.
 
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEy(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    
    '''
    calculates the crosstalk contribution to the global on sky (y) axis from a particular detector. the contribution is calculated by the trigonometry from the alignment of the detector, cross detector with the global x, y, a, b axes. 
    
    returns pixel maps with the new crosstalk contribution position and magnitude added
    
    Inputs:
    ****************************************************************************
    det [int]: the identifying number of the detector in question being crosstalked to
    det_dict [dictionary]: focal plane information storing detector number along with frequency response, angle relative to global sky coordinates, and signal magnitude
    coupling_dict [dictionary]: the crosstalk matrix that defines the detector to detector coupling
    freq1 [int]: first frequency being considered
    freq2 [int]: second frequency being considered, for crosstalk between adjacent bands these two numbers would be different
    perc_corr [float]: the percent to which the signals of the detectors are correlated
    empty_beam_map [2D array]: a copy of null beam maps to size the pixel beam maps appropriately
    pixel_size [float]: size of each pixel in the map in arcmin
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    pixel_map_x [2D array]: contribution of this detector and crosstalk signal along the global sky x axis
    pixel_map_y [2D array]: contribution of this detector and crosstalk signal along the global sky y axis
    pixel_map_a [2D array]: contribution of this detector and crosstalk signal along the global sky a axis
    pixel_map_b [2D array]: contribution of this detector and crosstalk signal along the global sky b axis
    ****************************************************************************
    '''
    
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()

    
    #crosstalk depends on relative orientation between detectors
    if coupling_dict[(freq1,freq2)][det] == 'na':
        crosstalk = 0
        
        i = 0
        j = 0
    else:
        #take coupled value and calculate crosstalk
        coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])

        #add to the relative offset in this pixel map
        i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
        j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  

    det_ang = det_dict[freq1][det]['ang']
    
    #make x, y, a, b maps
    #if aligned with global x, y        
    #for contribution to global x,y,a,b for pure Ex signal
    pixel_map_x[int(N/2),int(N/2)] = 0.
    pixel_map_y[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - np.pi/2.))**2.
    pixel_map_a[int(N/2),int(N/2)] = 0.
    pixel_map_b[int(N/2),int(N/2)] = 0.
    
    #add crosstalk signal projections
    pixel_map_x[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 0.))**2.
    pixel_map_y[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/2.))**2.
    pixel_map_a[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/4.))**2.
    pixel_map_b[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 3*np.pi/4))**2.
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEa(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    
    '''
    calculates the crosstalk contribution to the global on sky (a) axis from a particular detector. the contribution is calculated by the trigonometry from the alignment of the detector, cross detector with the global x, y, a, b axes. 
    
    returns pixel maps with the new crosstalk contribution position and magnitude added
    
    Inputs:
    ****************************************************************************
    det [int]: the identifying number of the detector in question being crosstalked to
    det_dict [dictionary]: focal plane information storing detector number along with frequency response, angle relative to global sky coordinates, and signal magnitude
    coupling_dict [dictionary]: the crosstalk matrix that defines the detector to detector coupling
    freq1 [int]: first frequency being considered
    freq2 [int]: second frequency being considered, for crosstalk between adjacent bands these two numbers would be different
    perc_corr [float]: the percent to which the signals of the detectors are correlated
    empty_beam_map [2D array]: a copy of null beam maps to size the pixel beam maps appropriately
    pixel_size [float]: size of each pixel in the map in arcmin
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    pixel_map_x [2D array]: contribution of this detector and crosstalk signal along the global sky x axis
    pixel_map_y [2D array]: contribution of this detector and crosstalk signal along the global sky y axis
    pixel_map_a [2D array]: contribution of this detector and crosstalk signal along the global sky a axis
    pixel_map_b [2D array]: contribution of this detector and crosstalk signal along the global sky b axis
    ****************************************************************************
    '''
    
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    

    #crosstalk depends on relative orientation between detectors
    if coupling_dict[(freq1,freq2)][det] == 'na':
        crosstalk = 0
        
        i = 0
        j = 0
    else:
        #take coupled value and calculate crosstalk
        coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
    
        #add to the relative offset in this pixel map
        i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
        j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  

    det_ang = det_dict[freq1][det]['ang']
    
    #make x, y, a, b maps
    #if aligned with global x, y        
    #for contribution to global x,y,a,b for pure Ex signal
    pixel_map_x[int(N/2),int(N/2)] = 0.
    pixel_map_y[int(N/2),int(N/2)] = 0.
    pixel_map_a[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - np.pi/4.))**2.
    pixel_map_b[int(N/2),int(N/2)] = 0.
    
    #add crosstalk signal projections
    pixel_map_x[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 0.))**2.
    pixel_map_y[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/2.))**2.
    pixel_map_a[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/4.))**2.
    pixel_map_b[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 3*np.pi/4))**2.
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEb(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    
    '''
    calculates the crosstalk contribution to the global on sky (b) axis from a particular detector. the contribution is calculated by the trigonometry from the alignment of the detector, cross detector with the global x, y, a, b axes. 
    
    returns pixel maps with the new crosstalk contribution position and magnitude added
    
    Inputs:
    ****************************************************************************
    det [int]: the identifying number of the detector in question being crosstalked to
    det_dict [dictionary]: focal plane information storing detector number along with frequency response, angle relative to global sky coordinates, and signal magnitude
    coupling_dict [dictionary]: the crosstalk matrix that defines the detector to detector coupling
    freq1 [int]: first frequency being considered
    freq2 [int]: second frequency being considered, for crosstalk between adjacent bands these two numbers would be different
    perc_corr [float]: the percent to which the signals of the detectors are correlated
    empty_beam_map [2D array]: a copy of null beam maps to size the pixel beam maps appropriately
    pixel_size [float]: size of each pixel in the map in arcmin
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    pixel_map_x [2D array]: contribution of this detector and crosstalk signal along the global sky x axis
    pixel_map_y [2D array]: contribution of this detector and crosstalk signal along the global sky y axis
    pixel_map_a [2D array]: contribution of this detector and crosstalk signal along the global sky a axis
    pixel_map_b [2D array]: contribution of this detector and crosstalk signal along the global sky b axis
    ****************************************************************************
    '''
    
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    

    #crosstalk depends on relative orientation between detectors
    if coupling_dict[(freq1,freq2)][det] == 'na':
        crosstalk = 0
        
        i = 0
        j = 0
    else:
        #take coupled value and calculate crosstalk
        coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])

        #add to the relative offset in this pixel map
        i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
        j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  

    det_ang = det_dict[freq1][det]['ang']
    
    #make x, y, a, b maps
    #if aligned with global x, y        
    #for contribution to global x,y,a,b for pure Ex signal
    pixel_map_x[int(N/2),int(N/2)] = 0.
    pixel_map_y[int(N/2),int(N/2)] = 0.
    pixel_map_a[int(N/2),int(N/2)] = 0.
    pixel_map_b[int(N/2),int(N/2)] = (det_dict[freq1][det]['sig'] * math.cos(det_ang - 3*np.pi/4.))**2.
    
    #add crosstalk signal projections
    pixel_map_x[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 0.))**2.
    pixel_map_y[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/2.))**2.
    pixel_map_a[i,j] = perc_corr*(crosstalk * math.cos(det_ang - np.pi/4.))**2.
    pixel_map_b[i,j] = perc_corr*(crosstalk * math.cos(det_ang - 3*np.pi/4))**2.
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b


 ###############################  ###############################  ############################### 
    
######################################Power Spectrum Modules################################
  
def get_IQU_fft(Imap, Qmap, Umap, to_plot=False):
    
    '''
    takes the 2D FFT of the focal plane I, Q, U maps and plots the auto and cross maps II, IQ, QU, etc
    
    returns the 2D FFT maps
    
    Inputs:
    ****************************************************************************
    Imap [2D array]: the I map as seen by the focal plane
    Qmap [2D array]: the Q map as seen by the focal plane
    Umap [2D array]: the U map as seen by the focal plane
    to_plot [boolean]: if True, plot the auto and cross maps, if False, do nothing
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    Imap_fft [2D array]: the 2D FFT of the I map
    Qmap_fft [2D array]: the 2D FFT of the Q map
    Umap_fft [2D array]: the 2D FFT of the U map
    ****************************************************************************
    '''
    
    
    #take fft's
    Imap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Imap)))
    Qmap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Qmap)))
    Umap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Umap)))
    
    
    if to_plot:
        #Plot beam maps in 3x3 matrix
        fig, ax = plt.subplots(1,3, figsize=(20,20))
        perc_corr = 0.03 #crosstalk at 3% level
        eta = 0.00001

        #II
        ax[0].imshow(10.*np.log(np.real(Imap_fft * np.conj(Imap_fft)) + eta))
        ax[0].set_title('I Map FFT')

        #IQ
        ax[1].imshow(10.*np.log(np.real(Qmap_fft * np.conj(Qmap_fft)) + eta))
        ax[1].set_title('Q Map FFT')

        #IU
        ax[2].imshow(10.*np.log(np.real(Umap_fft * np.conj(Umap_fft)) + eta))
        ax[2].set_title('U Map FFT')

    return Imap_fft,Qmap_fft,Umap_fft
    

def calculate_2d_spectra(Imap=None,Qmap=None,Umap=None,delta_ell=50,ell_max=5000,pix_size=0.25,N=1024):
    
    '''
    calculates the 2D FFT of the focal plane I, Q, U and converts these to T, E, B space.
    
    Inputs:
    ****************************************************************************
    Imap [2D array]: the I map detected by the focal plane
    Qmap [2D array]: the Q map detected by the focal plane
    Umap [2D array]: the U map detected by the focal plane
    delta_ell [int]: bin size in ell space 
    ell_max [int]: maximum ell for binning
    pix_size [float]: the size of each pixel in arcmin
    N [float]: number of pixels in a side of a map
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    maps_dict [dictionary of 2D array]: dictionary of 2D FFT maps keyed by TT, TE, EE, etc
    ****************************************************************************
    '''
    
    N=int(N)
    # make a 2d ell coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    
    #store maps in dict
    maps_dict = {}
    
    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array_TT = np.zeros(N_bins)
    CL_array_EE = np.zeros(N_bins)
    CL_array_BB = np.zeros(N_bins)
    CL_array_TE = np.zeros(N_bins)
    CL_array_EB = np.zeros(N_bins)
    CL_array_TB = np.zeros(N_bins)
    
    if Imap is not None:
        # get the 2d fourier transform of the map
        #fftshift to shift to center of map, then fourier transform, then fftshift to shift ell=0 to center
        Imap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Imap)))
        TTmap = np.real(np.conj(Imap_fft) * Imap_fft)
        maps_dict['TT'] = TTmap
        
    if Qmap is not None and Umap is not None:
        #fftshift to shift to center of map, then foureir transform, then fftshift to shift ell=0 to center
        Qmap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Qmap)))
        Umap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Umap)))
        
        #initialize EB FFT maps
        Emap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        Bmap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        
        #calculate fourier angle
        theta_ell = np.arctan2(kY,kX)
        
        #rotate convolved Q and U beams into E and B beams in fourier space
        #Emap = np.divide((Qmap_fft * np.cos(2 * theta_ell ) + Umap_fft * np.sin( 2 * theta_ell )), deproject_angle_E)
        #Bmap = np.divide((-Qmap_fft * np.sin(2 * theta_ell ) + Umap_fft * np.cos( 2 * theta_ell )), deproject_angle_B)
        Emap = (Qmap_fft * np.cos( 2 * theta_ell ) + Umap_fft * np.sin( 2 * theta_ell ))
        Bmap = (-Qmap_fft * np.sin( 2 * theta_ell ) + Umap_fft * np.cos( 2 * theta_ell ))
        
        #if deproject:
        #    assert(unconvolved_beams is not None)
        #    #FLAG: Deproject if delta-function input
        #    #Emap,Bmap = deproject_delta_function(Emap,Bmap,theta_ell)
        #    unconvolved_Imap_fft, unconvolved_Qmap_fft, unconvolved_Umap_fft = unconvolved_map_ffts(unconvolved_beams, sky_decomp)

        #    #deproject if both Q and U have the same beam at 1.5' fwhm
        #    inst_beam_1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(make_2d_gaussian_beam(N,pix_size, 1.5/60.))))
        #    Emap,Bmap = deproject_input_maps(Qmap_fft, Umap_fft, unconvolved_Qmap_fft, unconvolved_Umap_fft, theta_ell)
        
        #calculate auto spectra
        EEmap = abs( np.conj(Emap) * Emap )
        BBmap = abs( np.conj(Bmap) * Bmap )
        
        maps_dict['EE'] = EEmap
        maps_dict['BB'] = BBmap
 
    #cross correlation via the correlation theorem
    #these will be normalized later to leakage beams
    if Imap is not None and Qmap is not None and Umap is not None:
        #TEmap = np.fft.fftshift(abs( (Imap_fft * np.conj(Emap) ) ))
        #EBmap = np.fft.fftshift(abs( (Emap * np.conj(Bmap) ) ))
        #TBmap = np.fft.fftshift(abs( (Imap_fft * np.conj(Bmap) ) ))
        TEmap = (abs( (Imap_fft * np.conj(Emap) ) ))
        EBmap = (abs( (Emap * np.conj(Bmap) ) ))
        TBmap = (abs( (Imap_fft * np.conj(Bmap) ) ))
        
        maps_dict['TB'] = TBmap
        maps_dict['TE'] = TEmap
        maps_dict['EB'] = EBmap
    
    return maps_dict

def calculate_2d_leakage_beams(Imap=None, Qmap=None, Umap=None, Qmap_deproj=None, Umap_deproj=None, delta_ell=50,ell_max=5000,pix_size=0.25,N=1024):

    '''
    calculates the 2D FFT's of the detector I, Q, U maps, converts these to T, E, and B space, and deconvolves the delta function bias from the E and B maps
    
    *deconvolved the input E and B maps from the QU maps that aren't convolved with the instrument beam. IQU maps are the same as the above function but the beam matrix is the 3x3 IQU matrix before convolution with the instrument beam. This will be deconvolved from the Emap and Bmap to give just the leakage beam without the input map contributions.
    
    returns a dictionary of 2D arrays spectra
    
    Inputs:
    ****************************************************************************
    Imap [2D array]: the real space I map the focal plane detects
    Qmap [2D array]: the real space Q map the focal plane detects
    Umap [2D array]: the real space U map the focal plane detects
    Qmap_deproj [2D array]: the real space Q pixel maps the focal plane detects
    Umap_deproj [2D array]: the real space U pixel maps the focal plane detects
    delta_ell [int]: bin size for spectra calculation *deprecated
    ell_max [int]: maximum ell for spectra calculation *deprecated
    pix_size [float]: pixel size of the maps in arcmin *deprecated
    N [int]: number of pixels in map side
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    maps_dict [dictionary of 2D array]: dictionary of 2D ell space maps
    ****************************************************************************
    '''
    
    N=int(N)
    # make a 2d ell coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    
    #store maps in dict
    maps_dict = {}
    
    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array_TT = np.zeros(N_bins)
    CL_array_EE = np.zeros(N_bins)
    CL_array_BB = np.zeros(N_bins)
    CL_array_TE = np.zeros(N_bins)
    CL_array_EB = np.zeros(N_bins)
    CL_array_TB = np.zeros(N_bins)
    
    if Imap is not None:
        # get the 2d fourier transform of the map
        #fftshift to shift to center of map, then foureir transform, then fftshift to shift ell=0 to center
        Imap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Imap)))
        TTmap = np.real(np.conj(Imap_fft) * Imap_fft)
        maps_dict['TT'] = TTmap
        
    if Qmap is not None and Umap is not None:
        #fftshift to shift to center of map, then foureir transform, then fftshift to shift ell=0 to center
        Qmap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Qmap)))
        Umap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Umap)))
        if Qmap_deproj is not None and Umap_deproj is not None:
            Qmap_deproj_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Qmap_deproj)))
            Umap_deproj_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Umap_deproj)))
        
        #initialize EB FFT maps
        Emap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        Bmap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        Emap_deproj = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        Bmap_deproj = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        
        #calculate fourier angle
        theta_ell = np.arctan2(kY,kX)
        
        #rotate convolved Q and U beams into E and B beams in fourier space
        #Emap = np.divide((Qmap_fft * np.cos(2 * theta_ell ) + Umap_fft * np.sin( 2 * theta_ell )), deproject_angle_E)
        #Bmap = np.divide((-Qmap_fft * np.sin(2 * theta_ell ) + Umap_fft * np.cos( 2 * theta_ell )), deproject_angle_B)
        Emap = (Qmap_fft * np.cos( 2 * theta_ell ) + Umap_fft * np.sin( 2 * theta_ell ))
        Bmap = (-Qmap_fft * np.sin( 2 * theta_ell ) + Umap_fft * np.cos( 2 * theta_ell ))
        
        #deproject input maps
        if Qmap_deproj is not None and Umap_deproj is not None:
            Emap, Bmap = deconvolve_input_maps(Emap, Bmap, Qmap_deproj_fft, Umap_deproj_fft, theta_ell)    
        
        #calculate auto spectra
        EEmap = abs( np.conj(Emap) * Emap )
        BBmap = abs( np.conj(Bmap) * Bmap )
        
        maps_dict['EE'] = EEmap
        maps_dict['BB'] = BBmap
 
    #cross correlation via the correlation theorem
    #these will be normalized later to leakage beams
    if Imap is not None and Qmap is not None and Umap is not None:
        #TEmap = np.fft.fftshift(abs( (Imap_fft * np.conj(Emap) ) ))
        #EBmap = np.fft.fftshift(abs( (Emap * np.conj(Bmap) ) ))
        #TBmap = np.fft.fftshift(abs( (Imap_fft * np.conj(Bmap) ) ))
        TEmap = (abs( (Imap_fft * np.conj(Emap) ) ))
        EBmap = (abs( (Emap * np.conj(Bmap) ) ))
        TBmap = (abs( (Imap_fft * np.conj(Bmap) ) ))
        
        maps_dict['TB'] = TBmap
        maps_dict['TE'] = TEmap
        maps_dict['EB'] = EBmap
    
    return maps_dict

def bin_maps_to_1d(maps_dict, delta_ell=50, ell_max=5000, pix_size=0.25, N=1024, square_TT=False):
        
    '''
    bins 2D maps to 1D power spectra
    
    returns a dictionary of 1D spectra in ell space
    
    Inputs:
    ****************************************************************************
    maps_dict [dictionary of 2D array]: dictionary of TT, TE, EE, etc 2D maps in ell space
    delta_ell [int]: bin size for spectra
    ell_max [int]: maximum ell the calculation goes out to
    pix_size [float]: the pixel size of the maps in arcmin
    N [int]: number of pixels in a map
    square_TT [bool]: find the magnitude squared of TT. Use False if maps_dict already has
                      magnitude squared of TT (i.e. if maps_dict is the output of calculate_2d_spectra)
    
    To not produce nans, N must be even and N * pix_size >= 244.5
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    ell_array [1D array]: the array of bins that coincide with the power spectra
    Cl_spec_dict [dictionary of 1D array]: dictionary of band powers for TT, TE, EE, etc
    ****************************************************************************    
    '''
    
    N=int(N)
    # make a 2d ell coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    maps_dict_keys = maps_dict.keys()
    
    ell2d_flat_sorted = np.sort(ell2d.flat)
    min_delta_ell = int(max(np.diff(ell2d_flat_sorted))/2 + 1)
    delta_ell = max(delta_ell, min_delta_ell)
    print('delta_ell = ' + str(delta_ell))
    
    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array_TT = np.zeros(N_bins)
    CL_array_EE = np.zeros(N_bins)
    CL_array_BB = np.zeros(N_bins)
    CL_array_TE = np.zeros(N_bins)
    CL_array_EB = np.zeros(N_bins)
    CL_array_TB = np.zeros(N_bins)    
   
    # from calculate_2d_spectra, maps_dict['TT'] = np.real(np.conj(Imap_fft) * Imap_fft))
    # so if using calculate_2d_spectra leave square_TT=False
    maps_dict_squared = {}
    if square_TT:
        maps_dict_squared['TT'] = np.real(np.conj(maps_dict['TT'])*maps_dict['TT'])
    else:
        maps_dict_squared['TT'] = maps_dict['TT']
    
    # fill out the spectra
    i = 0
    while (i < N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array_TT[i] = np.mean(maps_dict_squared['TT'][inds_in_bin])
        
        if 'TT' in maps_dict_keys and 'EE' in maps_dict_keys:
            CL_array_EE[i] = np.mean(maps_dict['EE'][inds_in_bin])
            CL_array_BB[i] = np.mean(maps_dict['BB'][inds_in_bin])
            CL_array_TE[i] = np.mean(maps_dict['TE'][inds_in_bin])
            CL_array_EB[i] = np.mean(maps_dict['EB'][inds_in_bin])
            CL_array_TB[i] = np.mean(maps_dict['TB'][inds_in_bin])
        #print i, ell_array[i], inds_in_bin, CL_array[i]
        i = i + 1
    
    Cl_spec_dict = {}
    if 'TT' in maps_dict_keys:
        Cl_spec_dict['TT'] = CL_array_TT*np.sqrt(pix_size /60.* np.pi/180.)*2.
    if 'TT' in maps_dict_keys and 'EE' in maps_dict_keys:
        Cl_spec_dict['EE'] = CL_array_EE*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['BB'] = CL_array_BB*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['TE'] = CL_array_TE*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['EB'] = CL_array_EB*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['TB'] = CL_array_TB*np.sqrt(pix_size /60.* np.pi/180.)*2.
    # return the power spectrum and ell bins
    return(ell_array,Cl_spec_dict)
    
    
    
#def calculate_2d_spectra(Imap=None,Qmap=None,Umap=None,delta_ell=50,ell_max=5000,pix_size=0.25,N=1024):
#    import numpy as np
#    N=int(N)
#    # make a 2d ell coordinate system
#    ones = np.ones(N)
#    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
#    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
#    kY = np.transpose(kX)
#    K = np.sqrt(kX**2. + kY**2.)
#    ell_scale_factor = 2. * np.pi 
#    ell2d = K * ell_scale_factor
#    
#    # make an array to hold the power spectrum results
#    N_bins = int(ell_max/delta_ell)
#    ell_array = np.arange(N_bins)
#    CL_array_TT = np.zeros(N_bins)
#    CL_array_EE = np.zeros(N_bins)
#    CL_array_BB = np.zeros(N_bins)
#    CL_array_TE = np.zeros(N_bins)
#    CL_array_EB = np.zeros(N_bins)
#    CL_array_TB = np.zeros(N_bins)
#    
#    if Imap is not None:
#        # get the 2d fourier transform of the map
#        Imap_fft = np.fft.fft2(np.fft.fftshift(Imap))
#        #FMap2 = np.fft.fft2(np.fft.fftshift(Map2))
#        TTmap = np.fft.fftshift(np.real(np.conj(Imap_fft) * Imap_fft))
#        
#    if Qmap is not None and Umap is not None:
#        Qmap_fft = np.fft.fft2(np.fft.fftshift(Qmap))
#        Umap_fft = np.fft.fft2(np.fft.fftshift(Umap))
#        Emap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
#        Bmap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
#        
#        Emap = Qmap_fft * np.cos(2 * np.arctan2(kY,kX) ) + Umap_fft * np.sin( 2 * np.arctan2(kY,kX) )
#        Bmap = -Qmap_fft * np.sin(2 * np.arctan2(kY,kX) ) + Umap_fft * np.cos( 2 * np.arctan2(kY,kX) )
#        
#       EEmap = np.fft.fftshift(np.real( np.conj(Emap) * Emap ))
#        BBmap = np.fft.fftshift(np.real( np.conj(Bmap) * Bmap ))
##        
#   if Imap is not None and Qmap is not None and Umap is not None:
#        TEmap = np.fft.fftshift(np.real(Imap_fft * np.conj(Emap) + np.conj(Imap_fft) * Emap))
#        EBmap = np.fft.fftshift(np.real(Emap * np.conj(Bmap) + np.conj(Emap) * Bmap))
#        TBmap = np.fft.fftshift(np.real(Imap_fft * np.conj(Bmap) + np.conj(Imap_fft) * Bmap))
#        
#    # fill out the spectra
#    i = 0
#    while (i < N_bins):
#        ell_array[i] = (i + 0.5) * delta_ell
#        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
#        CL_array_TT[i] = np.mean(TTmap[inds_in_bin])
#        
#        if Imap is not None and Qmap is not None and Umap is not None:
#            CL_array_EE[i] = np.mean(EEmap[inds_in_bin])
#            CL_array_BB[i] = np.mean(BBmap[inds_in_bin])
#            CL_array_TE[i] = np.mean(TEmap[inds_in_bin])
#            CL_array_EB[i] = np.mean(EBmap[inds_in_bin])
#            CL_array_TB[i] = np.mean(TBmap[inds_in_bin])
#        #print i, ell_array[i], inds_in_bin, CL_array[i]
#        i = i + 1
# 
#    Cl_spec_dict = {}
#    if Imap is not None:
#        Cl_spec_dict['TT'] = CL_array_TT*np.sqrt(pix_size /60.* np.pi/180.)*2.
#    if Qmap is not None and Umap is not None:
#        Cl_spec_dict['EE'] = CL_array_EE*np.sqrt(pix_size /60.* np.pi/180.)*2.
#        Cl_spec_dict['BB'] = CL_array_BB*np.sqrt(pix_size /60.* np.pi/180.)*2.
#        Cl_spec_dict['TE'] = CL_array_TE*np.sqrt(pix_size /60.* np.pi/180.)*2.
#        Cl_spec_dict['EB'] = CL_array_EB*np.sqrt(pix_size /60.* np.pi/180.)*2.
#        Cl_spec_dict['TB'] = CL_array_TB*np.sqrt(pix_size /60.* np.pi/180.)*2.
#    # return the power spectrum and ell bins
#    return(ell_array,Cl_spec_dict)    
    
def deconvolve_delta_function(Emap, Bmap, theta_ell):
    
    '''
    deconvolves a delta function from E and B maps
    
    returns E and B maps
    
    Inputs:
    ****************************************************************************
    Emap [2D array]: the E map made from combining FFT's of Q and U maps
    Bmap [2D array]: the B map made from combining FFT's of Q and U maps
    theta_ell [2D array]: the angle in ell space defined by arctan(ly,lx)
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    Emap [2D array]: the E map with a delta function removed
    Bmap [2D array]: the B map with a delta function removed
    ****************************************************************************    
    '''
    
    deproject_E = np.sin(2*theta_ell) + np.cos(2*theta_ell)
    deproject_B = np.cos(2*theta_ell) - np.sin(2*theta_ell)
    
    Emap = np.divide(Emap, deproject_E)
    Bmap = np.divide(Bmap, deproject_B)
    
    return (Emap, Bmap) 

def deconvolve_beam(Qmap_fft, Umap_fft, theta_ell, inst_beam_1):
    
    '''
    deconvolves the instrument beam from the E and B maps
    
    returns a tuple of E and B maps with the instrument beams removed
    
    Inputs:
    ****************************************************************************
    Qmap_fft [2D array]: the 2D FFT of the Q map
    Umap_fft [2D array]: the 2D FFT of the U map
    theta_ell [2D array]: the angle in ell space defined by arctan(ly,lx)
    inst_beam_1 [2D array]: the 2D FFt of the instrument beam 
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    Emap [2D array]: the E map with the instrument beam removed
    Bmap [2D array]: the B map with the instrument beam removed
    ****************************************************************************
    '''
    
    #deconvolve beam to get true EB maps
    Emap_convolved = Qmap_fft * np.cos(2 * theta_ell) + Umap_fft * np.sin(2 * theta_ell)
    Bmap_convolved = -Qmap_fft * np.sin(2 * theta_ell) + Umap_fft * np.cos(2 * theta_ell)
    Emap = np.divide(Emap_convolved, inst_beam_1, out=np.zeros_like(Emap_convolved), where=inst_beam_1!=0)
    Bmap = np.divide(Bmap_convolved, inst_beam_1, out=np.zeros_like(Bmap_convolved), where=inst_beam_1!=0)
           
    
    return (Emap, Bmap)

def deconvolve_input_maps(Emap_fft, Bmap_fft, unconvolved_Qmap_fft, unconvolved_Umap_fft, theta_ell):
    
    '''
    deconvolves the pixelized Q and U maps from the E and B maps that contain delta function biases left over from taking a 2D fft of a convolved beam map
    
    returns the deconvolved E and B maps
    
    Inputs:
    ****************************************************************************
    Emap_fft [2D array]: the E map made from the pixelized Q and U maps with the instrument beams
    Bmap_fft [2D array]: the B map made from the pixelized Q and U maps with the instrument beams
    unconvolved_Qmap_fft [2D array]: the pixelized Q map FFT's 
    unconvolved_Umap_fft [2D array]: the pixelized U map FFT's
    theta_ell [2D array]: the angle in ell space defined by arctan(ly,lx)
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    Emap_dep [2D array]: the E map with the delta function bias removed
    Bmap_dep [2D array]: the B map with the detla function bias removed
    ****************************************************************************
    '''
    Emap_unconvolved = unconvolved_Qmap_fft*np.cos(2*theta_ell) + unconvolved_Umap_fft*np.sin(2*theta_ell)
    Bmap_unconvolved = -unconvolved_Qmap_fft*np.sin(2*theta_ell) + unconvolved_Umap_fft*np.cos(2*theta_ell)
    
    Emap_dep = np.divide(Emap_fft, Emap_unconvolved, out=np.zeros_like(Emap_fft), where=Emap_unconvolved!=0)
    Bmap_dep = np.divide(Bmap_fft, Bmap_unconvolved, out=np.zeros_like(Bmap_fft), where=Bmap_unconvolved!=0)
    
    
    return Emap_dep,Bmap_dep

def unconvolved_map_ffts(unconvolved_beams_dict, sky_decomp):
    
    '''
    Takes the 2D FFT of real space pixel maps that are not convolved with the instrument beam. This does not have the ability to contain a sky decomposition that are simulated CMB maps. The outputs are for removing delta function biases in the final results, especially for E and B maps.
    
    returns the pixelized 2D FFT's of the I, Q, U maps the focal plane detects
    
    Inputs:
    ****************************************************************************
    unconvolved_beams_dict [dictionary]: the on sky beam map of pixels that are not convolved with the instrument beam
    sky_decomp [1D list]: the linear decomposition of the sky in terms of unit vector in I, Q, U space
    ****************************************************************************
    
    Outputs:
    ****************************************************************************
    Imap_fft [2D array]: the pixelized I map the focal plane detects
    Qmap_fft [2D array]: the pixelized Q map the focal plane detects
    Umap_fft [2D array]: the pixelized U map the focal plane detects
    ****************************************************************************
    '''
    
    Imap = sky_decomp[0]*unconvolved_beams_dict['II'] + sky_decomp[1]*unconvolved_beams_dict['IQ'] + sky_decomp[2]*unconvolved_beams_dict['IU']
    Qmap = sky_decomp[0]*unconvolved_beams_dict['QI'] + sky_decomp[1]*unconvolved_beams_dict['QQ'] + sky_decomp[2]*unconvolved_beams_dict['QU']
    Umap = sky_decomp[0]*unconvolved_beams_dict['UI'] + sky_decomp[1]*unconvolved_beams_dict['UQ'] + sky_decomp[2]*unconvolved_beams_dict['UU']
    
    Imap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Imap)))
    Qmap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Qmap)))
    Umap_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Umap)))
    
    
    return Imap_fft,Qmap_fft,Umap_fft



 ###############################  ###############################  ############################### 
    
 ######################################## Return Outputs #########################################


def output_toml(path, filename, beam_matrix, binned_ell, binned_spectra_dict):
    
    import toml
    import os
    
    '''
    Outputs the 3x3 beam matrix, 1d spectra, and the ell array used for binning to a toml file for interfacing with other codes. The filename need not have the .toml extention
    
    returns NA
    
    Inputs:
    ****************************************************************************
    path [string]: the absolute file path where the toml file should be saved to
    filename [string]: the desired name of the saved output toml file
    beam_matrix [2D array]: the 3x3 beam coupling matrix of 2D beam maps in real space
    binned_ell [1D array]: the list of ell bins used to calculate spectra
    binned_spectra_dict [dictionary]: dictionary of 1D spectra keyed by the names TT, TE, EE, etc 
    ****************************************************************************
    
    Outputs:
    NA
    '''
    
    #collect data into dictionary
    toml_dict = {}
    toml_dict['beam_matrix'] = beam_matrix
    toml_dict['ell_array'] = binned_ell
    toml_dict['spectra'] = binned_spectra_dict
    
    toml_string = toml.dumps(toml_dict)

    os.chdir(path)
    
    #write to a toml file
    file = open(filename + str('.toml'),"w")
    a = file.write(toml_string)
    file.close()
    
    
    return




 ###############################  ###############################  ############################### 
    
