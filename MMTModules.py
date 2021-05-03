#############################################Simulation Modules###########################################

def calculate_crosstalk(det_dict, freqs, pixel_size, perc_corr, N, beam_fwhm, sky_decomp, TtoP_suppress, delta_ell, ell_max, choose_normalization):
    import numpy as np
    import matplotlib.pyplot as plt
    
    #frequency under question
    freq1 = freqs[0]
    #frequency correlated to freq1
    freq2 = freqs[1]
    
    #run a simulation
    beam_matrix = calculate_beam_matrix(det_dict, freq1, freq2, pixel_size, perc_corr, N, TtoP_suppress = TtoP_suppress)
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
    inst_beam_1 = make_2d_gaussian_beam(N, pixel_size, beam_fwhm)
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


    #Generate 1D power spectra from beam maps
    pix_size = pixel_size * 60. #pixel size in arcmin
    Imap = sky_decomp[0] * convolved_coupled_beams['II'] + sky_decomp[1] * convolved_coupled_beams['IQ'] + sky_decomp[2] * convolved_coupled_beams['IU']
    Qmap = sky_decomp[0] * convolved_coupled_beams['QI'] + sky_decomp[1] * convolved_coupled_beams['QQ'] + sky_decomp[2] * convolved_coupled_beams['QU']
    Umap = sky_decomp[0] * convolved_coupled_beams['UI'] + sky_decomp[1] * convolved_coupled_beams['UQ'] + sky_decomp[2] * convolved_coupled_beams['UU']
    binned_ell, binned_spectra_dict = calculate_2d_spectra(Imap=Imap, Qmap=Qmap, Umap=Umap, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)
    binned_ell, beam_spectrum = calculate_2d_spectra(Imap=inst_beam_1, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N)


    
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
        if choose_normalization is 'TT':
            norm_fac = np.max(binned_spectra_dict['TT'][1:])
        elif choose_normalization is 'EE':
            norm_fac = np.max(binned_spectra_dict['EE'][1:])
        elif choose_normalization is 'BB':
            norm_fac = np.max(binned_spectra_dict['BB'][1:])
        else:
            print('Please use either TT, EE, or BB keys to study leakage effects')

        beam_spectrum['TT'] = beam_spectrum['TT'][1:] / np.max(beam_spectrum['TT'][1:])
        binned_spectra_dict['TT'] = binned_spectra_dict['TT'][1:] / norm_fac
        binned_spectra_dict['EE'] = binned_spectra_dict['EE'][1:] / norm_fac
        binned_spectra_dict['BB'] = binned_spectra_dict['BB'][1:] / norm_fac
        binned_spectra_dict['TE'] = binned_spectra_dict['TE'][1:] / norm_fac
        binned_spectra_dict['EB'] = binned_spectra_dict['EB'][1:] / norm_fac
        binned_spectra_dict['TB'] = binned_spectra_dict['TB'][1:] / norm_fac


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
        cross_labels = ['T->E','E->B','T->B']
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



    #return beam maps and raw power spectra
    return convolved_coupled_beams, binned_ell, binned_spectra_dict

def calculate_beam_matrix(det_dict, freq1, freq2, pixel_size, perc_corr, N, TtoP_suppress = False):
    import numpy as np
    import time
   
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
    coupling_dict = generate_random_coupling(det_dict)
    
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
    IQ_1 = (pixel_map_x_total + pixel_map_y_total) / (num_det_x * norm_fac / num_det)
    QQ_1 = (pixel_map_x_total - pixel_map_y_total) / (num_det_x * norm_fac / num_det)
    UQ_1 = (pixel_map_a_total - pixel_map_b_total) / (num_det_x * norm_fac / num_det)
    
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
    IQ_2 = (pixel_map_x_total + pixel_map_y_total) / (num_det_y * norm_fac / num_det)
    QQ_2 = (pixel_map_x_total - pixel_map_y_total) / (num_det_y * norm_fac / num_det)
    UQ_2 = (pixel_map_a_total - pixel_map_b_total) / (num_det_y * norm_fac / num_det)
    
    IQ = (IQ_1 + IQ_2) / 2.
    QQ = (np.abs(QQ_1) + np.abs(QQ_2)) / 2.
    UQ = (UQ_1 + UQ_2) / 2.
    
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
    IU_1 = (pixel_map_x_total + pixel_map_y_total) / (num_det_a * norm_fac / num_det)
    QU_1 = (pixel_map_x_total - pixel_map_y_total) / (num_det_a * norm_fac / num_det)
    UU_1 = (pixel_map_a_total - pixel_map_b_total) / (num_det_a * norm_fac / num_det)
    
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
    IU_2 = (pixel_map_x_total + pixel_map_y_total) / (num_det_b * norm_fac / num_det)
    QU_2 = (pixel_map_x_total - pixel_map_y_total) / (num_det_b * norm_fac / num_det)
    UU_2 = (pixel_map_a_total - pixel_map_b_total) / (num_det_b * norm_fac / num_det)
    
    IU = (IU_1 + IU_2) / 2.
    QU = (QU_1 + QU_2) / 2.
    UU = (np.abs(UU_1) + np.abs(UU_2)) / 2.
    
    #Artificially suppress U->I coupling to avoid double counting
    IU[int(N/2.),int(N/2.)] = 0.
    
    beam_keys = ['II','IQ','IU','QI','QQ','QU','UI','UQ','UU']
    beams = [II,IQ,IU,QI,QQ,QU,UI,UQ,UU]
    
    beam_matrix = {}
    for i in range(len(beam_keys)):
        beam_matrix[beam_keys[i]] = beams[i]
    
    return beam_matrix

def generate_random_coupling(det_dict):
    import numpy.random as random
   
    
    freqs = det_dict.keys()
    coupling_dict = {}
    for freq1 in freqs:
        for freq2 in freqs:
            coupling_dict[(freq1,freq2)] = {}
            for det in det_dict[freq1].keys():
                coupling_dict[(freq1,freq2)][det] = random.randint(0,len(det_dict[freq1].keys())-1)

    
    return coupling_dict

#FIXIT: Path to positions should be arbitrary
def generate_focal_plane_distribution(path_to_positions, num_det, freqs, rescale):
    import pandas as pd
    import numpy as np
    import random
    
    #Load spatial data

    #load from csv
    det_pos = pd.read_csv(path_to_positions,sep = ' ',names = ['x','y'])

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

            if orientation is 1:
                det_dict[freq1][det]['ang'] = 0.
                #det_dict[det]['angy'] = 0.

            elif orientation is 2:
                det_dict[freq1][det]['ang'] = np.pi / 4.
                #det_dict[det]['angy'] = np.pi / 4.

    #assign detector taus
    #define detector time constants (sampling from a gamma distribution)
    det_dict = sample_det_tau(det_dict)

    return det_dict

 ###############################  ###############################  ############################### 

#############################################Beam Modules################################################
def convolve_pixel_instrument(beam_map, inst_beam_map):
    import numpy as np
    #fft of beam_map
    beam_map_fft = np.fft.fft2(np.fft.fftshift(beam_map))
    
    #fft of inst_beam_map
    inst_beam_map_fft = np.fft.fft2(np.fft.fftshift(inst_beam_map))
    
    #convolve
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(beam_map_fft * inst_beam_map_fft)))
       
    return convolved_map
 

def make_2d_gaussian_beam(N,pix_size,beam_size_fwhp):
    import numpy as np
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
    import numpy as np
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
    import math
    import numpy as np
    #number of pixels on each map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    
    #take coupled value and calculate crosstalk
    coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
    
    #crosstalk depends on relative orientation between detectors
    if np.abs(det_dict[freq1][det]['ang'] - coupled_det['ang']) == 0.:
        if det % 2. == 0. and coupling_dict[(freq1,freq2)][det] % 2. == 0.:
            crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
        elif det % 2. != 0. and coupling_dict[(freq1,freq2)][det] % 2. != 0.:
            crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
        else:
            #crosstalk is zero if detectors are orthogonal
            crosstalk = 0.
    else:
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
    
    #add to the relative offset in this pixel map
    i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
    j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  
    #make x, y, a, b maps    
    #if aligned with global x, y        
    if det_dict[freq1][det]['ang'] == 0.:
         
        if det % 2 == 0:
            #set central x value
            pixel_map_x[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            pixel_map_y[int(N/2),int(N/2)] = 0.
            
            #add crosstalk at i and j
            #for global x detector
            pixel_map_x[i,j] = perc_corr * crosstalk**2.
            pixel_map_y[i,j] = 0.
            
            #central value of a and b maps for x detector
            pixel_map_a[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_b[int(N/2),int(N/2)] = ( math.sin( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            #set central y value
            pixel_map_x[int(N/2),int(N/2)] = 0.
            pixel_map_y[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            
            #add crosstalk at i and j
            #for global y detector
            pixel_map_x[i,j] = 0.
            pixel_map_y[i,j] = perc_corr * crosstalk**2.

            #projected a,b map
            pixel_map_a[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_b[int(N/2),int(N/2)] = ( math.sin( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
    
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
    elif det_dict[freq1][det]['ang'] == np.pi / 4.:
        #make a,b map
        
        if det % 2 == 0:
            pixel_map_a[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            pixel_map_b[int(N/2),int(N/2)] = 0.
            
            pixel_map_a[i,j] = perc_corr * crosstalk**2.
            pixel_map_b[i,j] = 0.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.sin( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            pixel_map_a[int(N/2),int(N/2)] = 0.
            pixel_map_b[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            
            pixel_map_a[i,j] = 0.
            pixel_map_b[i,j] = perc_corr * crosstalk**2.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.sin( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
        
    else:
        print('Not here')
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEx(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    import numpy as np
    import math
    
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    
    #take coupled value and calculate crosstalk
    coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
    
    #crosstalk depends on relative orientation between detectors
    if coupled_det['ang'] == 0.:
        if coupling_dict[(freq1,freq2)][det] % 2 == 0.:
            crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig']) 
        else:
            crosstalk = 0.
    else:
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
    
    
    #add to the relative offset in this pixel map
    i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
    j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  
    #make x, y, a, b maps
    #if aligned with global x, y        
    if det_dict[freq1][det]['ang'] == 0.:
         
        if det % 2 == 0:
            #set central x value
            pixel_map_x[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            pixel_map_y[int(N/2),int(N/2)] = 0.
            
            #add crosstalk at i and j
            #for global x detector
            pixel_map_x[i,j] = perc_corr * crosstalk**2.
            pixel_map_y[i,j] = 0.
            
            #central value of a and b maps for x detector
            pixel_map_a[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_b[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            #set central y value
            pixel_map_x[int(N/2),int(N/2)] = 0.
            pixel_map_y[int(N/2),int(N/2)] = 0.
            
            
            #add crosstalk at i and j
            #for global y detector
            pixel_map_x[i,j] = 0.
            pixel_map_y[i,j] = perc_corr * crosstalk**2.

            #projected a,b map
            pixel_map_a[int(N/2),int(N/2)] = 0.
            pixel_map_b[int(N/2),int(N/2)] = 0.
    
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
            
    elif det_dict[freq1][det]['ang'] == np.pi / 4.:
        #make a,b map
        
        if det % 2 == 0:
            pixel_map_a[int(N/2),int(N/2)] = math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            pixel_map_b[int(N/2),int(N/2)] = math.sin(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            
            pixel_map_a[i,j] = perc_corr * crosstalk**2.
            pixel_map_b[i,j] = 0.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            pixel_map_a[int(N/2),int(N/2)] = math.sin(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            pixel_map_b[int(N/2),int(N/2)] = math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            
            pixel_map_a[i,j] = 0.
            pixel_map_b[i,j] = perc_corr * crosstalk**2.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
        
    else:
        print('Not here')
 
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEy(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    import numpy as np
    import math
    
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    
    #take coupled value and calculate crosstalk
    coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
    
    #crosstalk depends on relative orientation between detectors
    if coupled_det['ang'] == 0.:
        if coupling_dict[(freq1,freq2)][det] % 2 == 0.:
            crosstalk = 0. 
        else:
            crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
    else:
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
    
    
    #add to the relative offset in this pixel map
    i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
    j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  
    #make x, y, a, b maps
    #if aligned with global x, y        
    if det_dict[freq1][det]['ang'] == 0.:
         
        if det % 2 == 0:
            #set central x value
            pixel_map_x[int(N/2),int(N/2)] = 0.
            pixel_map_y[int(N/2),int(N/2)] = 0.
            
            #add crosstalk at i and j
            #for global x detector
            pixel_map_x[i,j] = perc_corr * crosstalk**2.
            pixel_map_y[i,j] = 0.
            
            #central value of a and b maps for x detector
            pixel_map_a[int(N/2),int(N/2)] = 0.
            pixel_map_b[int(N/2),int(N/2)] = 0.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            #set central y value
            pixel_map_x[int(N/2),int(N/2)] = 0.
            pixel_map_y[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            
            
            #add crosstalk at i and j
            #for global y detector
            pixel_map_x[i,j] = 0.
            pixel_map_y[i,j] = perc_corr * crosstalk**2.

            #projected a,b map
            pixel_map_a[int(N/2),int(N/2)] = ( math.sin( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_b[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
    
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
            
    elif det_dict[freq1][det]['ang'] == np.pi / 4.:
        #make a,b map
        
        if det % 2 == 0:
            pixel_map_a[int(N/2),int(N/2)] = math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            pixel_map_b[int(N/2),int(N/2)] = math.sin(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            
            pixel_map_a[i,j] = perc_corr * crosstalk**2.
            pixel_map_b[i,j] = 0.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            pixel_map_a[int(N/2),int(N/2)] = math.sin(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            pixel_map_b[int(N/2),int(N/2)] = math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig']
            
            pixel_map_a[i,j] = 0.
            pixel_map_b[i,j] = perc_corr * crosstalk**2.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
        
    else:
        print('Not here')
    
    
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEa(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    import math
    import numpy as np
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    
    #take coupled value and calculate crosstalk
    coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
    
    #crosstalk depends on relative orientation between detectors
    if coupled_det['ang'] == np.pi / 4.:
        if coupling_dict[(freq1,freq2)][det] % 2. == 0.:
            crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
        else:
            crosstalk = 0.
    else:
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
  
    #add to the relative offset in this pixel map
    i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
    j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  
    #make x, y, a, b maps
    #if aligned with global x, y        
    if det_dict[freq1][det]['ang'] == 0.:
         
        if det % 2 == 0:
            #set central x value
            pixel_map_x[int(N/2),int(N/2)] = (math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig'])
            pixel_map_y[int(N/2),int(N/2)] = (math.sin(np.pi/4.)**2. * det_dict[freq1][det]['sig'])
            
            #add crosstalk at i and j
            #for global x detector
            pixel_map_x[i,j] = perc_corr * crosstalk**2.
            pixel_map_y[i,j] = 0.
            
            #central value of a and b maps for x detector
            pixel_map_a[int(N/2),int(N/2)] = (math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig'])**2.
            pixel_map_b[int(N/2),int(N/2)] = (math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig'])**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            #set central y value
            pixel_map_x[int(N/2),int(N/2)] = (math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig'])
            pixel_map_y[int(N/2),int(N/2)] = (math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig'])
            
            #add crosstalk at i and j
            #for global y detector
            pixel_map_x[i,j] = 0.
            pixel_map_y[i,j] = perc_corr * crosstalk**2.

            #projected a,b map
            pixel_map_a[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            pixel_map_b[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
    elif det_dict[freq1][det]['ang'] == np.pi / 4.:
        #make a,b map
        
        if det % 2 == 0:
            pixel_map_a[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            pixel_map_b[int(N/2),int(N/2)] = 0.
            
            pixel_map_a[i,j] = perc_corr * crosstalk**2.
            pixel_map_b[i,j] = 0.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.sin( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            pixel_map_a[int(N/2),int(N/2)] = 0.
            pixel_map_b[int(N/2),int(N/2)] = 0.
            
            pixel_map_a[i,j] = 0.
            pixel_map_b[i,j] = perc_corr * crosstalk**2.
            
            pixel_map_x[int(N/2),int(N/2)] = 0.
            pixel_map_y[int(N/2),int(N/2)] = 0.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
        
    else:
        print('Not here')
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

def crosstalk_pix_map_pureEb(det, det_dict, coupling_dict, freq1, freq2, perc_corr, empty_beam_map, pixel_size):
    import math
    import numpy as np
    #number of pixels along map edge
    N = len(empty_beam_map[0])
    
    #initialize pixel maps
    pixel_map_x = empty_beam_map.copy()
    pixel_map_y = empty_beam_map.copy()
    pixel_map_a = empty_beam_map.copy()
    pixel_map_b = empty_beam_map.copy()
    
    #take coupled value and calculate crosstalk
    coupled_det = det_dict[freq1][coupling_dict[(freq1,freq2)][det]]
    
    #crosstalk depends on relative orientation between detectors
    if coupled_det['ang'] == np.pi / 4.:
        if coupling_dict[(freq1,freq2)][det] % 2. == 0.:
            crosstalk = 0.
        else:
            crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
    else:
        crosstalk = ( math.cos(det_dict[freq1][det]['ang'] - coupled_det['ang']) * coupled_det['sig'])
  
    #add to the relative offset in this pixel map
    i = int(coupled_det['x']/pixel_size + N/2 - det_dict[freq1][det]['x']/pixel_size)
    j = int(coupled_det['y']/pixel_size + N/2 - det_dict[freq1][det]['y']/pixel_size)
  
    #make x, y, a, b maps
    #if aligned with global x, y        
    if det_dict[freq1][det]['ang'] == 0.:
         
        if det % 2 == 0:
            #set central x value
            pixel_map_x[int(N/2),int(N/2)] = (math.cos(np.pi/4.) * det_dict[freq1][det]['sig'])**2.
            pixel_map_y[int(N/2),int(N/2)] = (math.sin(np.pi/4.) * det_dict[freq1][det]['sig'])**2.
            
            #add crosstalk at i and j
            #for global x detector
            pixel_map_x[i,j] = perc_corr * crosstalk**2.
            pixel_map_y[i,j] = 0.
            
            #central value of a and b maps for x detector
            pixel_map_a[int(N/2),int(N/2)] = (math.cos(np.pi/4.)**2. * det_dict[freq1][det]['sig'])**2.
            pixel_map_b[int(N/2),int(N/2)] = (math.sin(np.pi/4.)**2. * det_dict[freq1][det]['sig'])**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            #set central y value
            pixel_map_x[int(N/2),int(N/2)] = (math.cos(np.pi/4.) * det_dict[freq1][det]['sig'])**2.
            pixel_map_y[int(N/2),int(N/2)] = (math.sin(np.pi/4.) * det_dict[freq1][det]['sig'])**2.
            
            #add crosstalk at i and j
            #for global y detector
            pixel_map_x[i,j] = 0.
            pixel_map_y[i,j] = perc_corr * crosstalk**2.

            #projected a,b map
            pixel_map_a[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
            pixel_map_b[int(N/2),int(N/2)] = ( math.cos( np.pi/4. )**2. * det_dict[freq1][det]['sig'] )**2.
    
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_a[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_a[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_b[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
    elif det_dict[freq1][det]['ang'] == np.pi / 4.:
        #make a,b map
        
        if det % 2 == 0:
            pixel_map_a[int(N/2),int(N/2)] = 0.
            pixel_map_b[int(N/2),int(N/2)] = 0.
            
            pixel_map_a[i,j] = perc_corr * crosstalk**2.
            pixel_map_b[i,j] = 0.
            
            pixel_map_x[int(N/2),int(N/2)] = 0.
            pixel_map_y[int(N/2),int(N/2)] = 0.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            
        else:
            pixel_map_a[int(N/2),int(N/2)] = 0.
            pixel_map_b[int(N/2),int(N/2)] = det_dict[freq1][det]['sig']
            
            pixel_map_a[i,j] = 0.
            pixel_map_b[i,j] = perc_corr * crosstalk**2.
            
            pixel_map_x[int(N/2),int(N/2)] = ( math.cos( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            pixel_map_y[int(N/2),int(N/2)] = ( math.sin( np.pi/4. ) * det_dict[freq1][det]['sig'] )**2.
            
            #if even index => detector oriented along x-y, if odd index detector oriented along a-b
            if coupling_dict[(freq1,freq2)][det] % 2 == 0:
                pixel_map_x[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
            else:
                pixel_map_x[i,j] = perc_corr * ( math.sin( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
                pixel_map_y[i,j] = perc_corr * ( math.cos( coupled_det['ang'] - np.pi/4. ) * crosstalk )**2.
        
        
    else:
        print('Not here')
    
    
    return pixel_map_x, pixel_map_y, pixel_map_a, pixel_map_b

 ###############################  ###############################  ############################### 
    
######################################Power Spectrum Modules################################
    
def calculate_2d_spectra(Imap=None,Qmap=None,Umap=None,delta_ell=50,ell_max=5000,pix_size=0.25,N=1024):
    import numpy as np
    N=int(N)
    # make a 2d ell coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    
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
        Imap_fft = np.fft.fft2(np.fft.fftshift(Imap))
        #FMap2 = np.fft.fft2(np.fft.fftshift(Map2))
        TTmap = np.fft.fftshift(np.real(np.conj(Imap_fft) * Imap_fft))
        
    if Qmap is not None and Umap is not None:
        Qmap_fft = np.fft.fft2(np.fft.fftshift(Qmap))
        Umap_fft = np.fft.fft2(np.fft.fftshift(Umap))
        Emap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        Bmap = np.zeros((len(Qmap_fft[:,1]),len(Qmap_fft[1,:])))
        
        Emap = Qmap_fft * np.cos(2 * np.arctan2(kY,kX) ) + Umap_fft * np.sin( 2 * np.arctan2(kY,kX) )
        Bmap = -Qmap_fft * np.sin(2 * np.arctan2(kY,kX) ) + Umap_fft * np.cos( 2 * np.arctan2(kY,kX) )
        
        EEmap = np.fft.fftshift(np.real( np.conj(Emap) * Emap ))
        BBmap = np.fft.fftshift(np.real( np.conj(Bmap) * Bmap ))
        
    if Imap is not None and Qmap is not None and Umap is not None:
        TEmap = np.fft.fftshift(np.real(Imap_fft * np.conj(Emap) + np.conj(Imap_fft) * Emap))
        EBmap = np.fft.fftshift(np.real(Emap * np.conj(Bmap) + np.conj(Emap) * Bmap))
        TBmap = np.fft.fftshift(np.real(Imap_fft * np.conj(Bmap) + np.conj(Imap_fft) * Bmap))
        
    # fill out the spectra
    i = 0
    while (i < N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array_TT[i] = np.mean(TTmap[inds_in_bin])
        
        if Imap is not None and Qmap is not None and Umap is not None:
            CL_array_EE[i] = np.mean(EEmap[inds_in_bin])
            CL_array_BB[i] = np.mean(BBmap[inds_in_bin])
            CL_array_TE[i] = np.mean(TEmap[inds_in_bin])
            CL_array_EB[i] = np.mean(EBmap[inds_in_bin])
            CL_array_TB[i] = np.mean(TBmap[inds_in_bin])
        #print i, ell_array[i], inds_in_bin, CL_array[i]
        i = i + 1
 
    Cl_spec_dict = {}
    if Imap is not None:
        Cl_spec_dict['TT'] = CL_array_TT*np.sqrt(pix_size /60.* np.pi/180.)*2.
    if Qmap is not None and Umap is not None:
        Cl_spec_dict['EE'] = CL_array_EE*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['BB'] = CL_array_BB*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['TE'] = CL_array_TE*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['EB'] = CL_array_EB*np.sqrt(pix_size /60.* np.pi/180.)*2.
        Cl_spec_dict['TB'] = CL_array_TB*np.sqrt(pix_size /60.* np.pi/180.)*2.
    # return the power spectrum and ell bins
    return(ell_array,Cl_spec_dict)    
    
 ###############################  ###############################  ############################### 

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

 ###############################  ###############################  ############################### 