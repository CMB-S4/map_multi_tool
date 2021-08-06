def generate_fiducial(ombh2 = 0.022, omch2 = 0.12, H0 = 68,
                     tau = 0.07, As = 2.2e-9, ns = 0.96, 
                     mnu = 0.06, nnu = 3.046, lmax = 5100):
    
    '''
    Generates a Cobaya fiducial TT power spectrum beginning at l=10
    and ending at lmax. This spectrum can be provided to the rest of
    the analysis functions. The inputs are LCDM parameters and the lmax
    '''
    
    #FIXIT: Do I need to modify this? Put in Inputs?
    packages_path = '/path/to/your/packages'
    
    #Using cobaya provider to provide fiducial
    fiducial_params = {
        'ombh2': ombh2, 'omch2': omch2, 'H0': H0, 'tau': tau,
        'As': As, 'ns': ns,
        'mnu': mnu, 'nnu': nnu}


    info_fiducial = {
        'params': fiducial_params,
        'likelihood': {'one': None},
        'theory': {'camb': {"extra_args": {"num_massive_neutrinos": 1}}},
        'packages_path': packages_path}


    from cobaya.model import get_model
    model_fiducial = get_model(info_fiducial)

    model_fiducial.add_requirements({"Cl": {'tt': lmax}})
    model_fiducial.logposterior({})
    Cls = model_fiducial.provider.get_Cl(ell_factor=False, units="muK2")
    
    Cl_fid = {}
    Cl_fid['TT'] = Cls['tt'][10:lmax+1]
    Cl_fid['TE'] = Cls['te'][10:lmax+1]
    Cl_fid['EE'] = Cls['ee'][10:lmax+1]
    

    
    return Cl_fid



def initialize_simulation(Nl=None, TF_eff=None, As_lower=1e-9, As_upper=4e-9, As_fid = 2.2e-9, ns_lower=0.9, ns_upper=1.1, ns_fid=0.96, nnu_lower=3.04, nnu_upper=3.05, nnu_fid=3.046):
    
    info = {
            'params': {
                # Fixed
                'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
                'mnu': 0.06,
                # Sampled
                'As': {'prior': {'min': As_lower, 'max': As_upper}, 'latex': 'A_s'},
                'ns': {'prior': {'min': ns_lower, 'max': ns_upper}, 'latex': 'n_s'},
                'nnu': {'prior': {'min': nnu_lower, 'max': nnu_upper}, 'latex': 'nnu'},
                # Derived
                'Map_Cl_at_500': {'latex': r'C_{500,\,\mathrm{map}}'}},
            'likelihood': {'my_cl_like': {
                "external": my_like_multi_spectra,
                # Declare required quantities!
                "requires": {'Cl': {'tt': lmax}},
                # Declare derived parameters!
                "output_params": ['Map_Cl_at_500']}},
            'theory': {'camb': {'stop_at_error': True}},
            'packages_path': packages_path}
    
    
    return info

def initialize_simulation_TT(Nl=None, TF_eff=None, As_lower=1e-9, As_upper=4e-9, As_fid = 2.2e-9, ns_lower=0.9, ns_upper=1.1, ns_fid=0.96, nnu_lower=3.04, nnu_upper=3.05, nnu_fid=3.046):
    
    #The below defines the simulation.
    #Sampled parameters As,ns,nnu
    if Nl is not None and TF_eff is None:

        info = {
            'params': {
                # Fixed
                'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
                'mnu': 0.06,
                # Sampled
                'As': {'prior': {'min': As_lower, 'max': As_upper}, 'latex': 'A_s'},
                'ns': {'prior': {'min': ns_lower, 'max': ns_upper}, 'latex': 'n_s'},
                'nnu': {'prior': {'min': nnu_lower, 'max': nnu_upper}, 'latex': 'nnu'},
                # Derived
                'Map_Cl_at_500': {'latex': r'C_{500,\,\mathrm{map}}'}},
            'likelihood': {'my_cl_like': {
                "external": my_like_noise,
                # Declare required quantities!
                "requires": {'Cl': {'tt': lmax}},
                # Declare derived parameters!
                "output_params": ['Map_Cl_at_500']}},
            'theory': {'camb': {'stop_at_error': True}},
            'packages_path': packages_path}

    elif Nl is None and TF_eff is not None:

        info = {
        'params': {
            # Fixed
            'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
            'mnu': 0.06,
            # Sampled
            'As': {'prior': {'min': As_lower, 'max': As_upper}, 'latex': 'A_s'},
            'ns': {'prior': {'min': ns_lower, 'max': ns_upper}, 'latex': 'n_s'},
            'nnu': {'prior': {'min': nnu_lower, 'max': nnu_upper}, 'latex': 'nnu'},
            # Derived
            'Map_Cl_at_500': {'latex': r'C_{500,\,\mathrm{map}}'}},
        'likelihood': {'my_cl_like': {
            "external": my_like_TF,
            # Declare required quantities!
            "requires": {'Cl': {'tt': lmax}},
            # Declare derived parameters!
            "output_params": ['Map_Cl_at_500']}},
        'theory': {'camb': {'stop_at_error': True}},
        'packages_path': packages_path}

    elif Nl is not None and TF_eff is not None:

        info = {
        'params': {
            # Fixed
            'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
            'mnu': 0.06,
            # Sampled
            'As': {'prior': {'min': As_lower, 'max': As_upper}, 'latex': 'A_s'},
            'ns': {'prior': {'min': ns_lower, 'max': ns_upper}, 'latex': 'n_s'},
            'nnu': {'prior': {'min': nnu_lower, 'max': nnu_upper}, 'latex': 'nnu'},
            # Derived
            'Map_Cl_at_500': {'latex': r'C_{500,\,\mathrm{map}}'}},
        'likelihood': {'my_cl_like': {
            "external": my_like_noise_TF,
            # Declare required quantities!
            "requires": {'Cl': {'tt': lmax}},
            # Declare derived parameters!
            "output_params": ['Map_Cl_at_500']}},
        'theory': {'camb': {'stop_at_error': True}},
        'packages_path': packages_path}

    else:

        info = {
        'params': {
            # Fixed
            'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
            'mnu': 0.06,
            # Sampled
            'As': {'prior': {'min': As_lower, 'max': As_upper}, 'latex': 'A_s'},
            'ns': {'prior': {'min': ns_lower, 'max': ns_upper}, 'latex': 'n_s'},
            'nnu': {'prior': {'min': nnu_lower, 'max': nnu_upper}, 'latex': 'nnu'},
            # Derived
            'Map_Cl_at_500': {'latex': r'C_{500,\,\mathrm{map}}'}},
        'likelihood': {'my_cl_like': {
            "external": my_like,
            # Declare required quantities!
            "requires": {'Cl': {'tt': lmax}},
            # Declare derived parameters!
            "output_params": ['Map_Cl_at_500']}},
        'theory': {'camb': {'stop_at_error': True}},
        'packages_path': packages_path}

    
    return info

def calculate_shift(model, nnu_lower, nnu_upper, num_nnu_samples, nnu_fid, As_fid, ns_fid):
    import numpy as np
    from cobaya.model import get_model
    
    
    '''
    Takes in a simulated Cobaya model and tries various nnu values in range to find the new maximum, which has been shifted from the fiducial by whatever effects are being studied. The function returns the shift in the nnu parameter space.
    '''

    ####################apply transfer functions to fiducial spectrum and use this power spectrum with cobaya##################

    #############################################################run model######################################################

    ############################sample over interval to back out liklihood function and find maximum############################

    # Plot of (prpto) probability density
    nnu = np.linspace(nnu_lower, nnu_upper, num_nnu_samples)
    loglikes_nnu = [model.loglike({'As':As_fid,'nnu': n,'ns':ns_fid})[0] for n in nnu]

    #######################################################take difference with fiducial central value###########################

    for i in range(len(loglikes_nnu)):
        if loglikes_nnu[i] == np.max(loglikes_nnu):
            shifted_nnu_max = nnu[i]

    print('nnu has been shifted by %s by noise and systematics.'%(np.abs(shifted_nnu_max - nnu_fid)))
    print('The new maximum liklihood is given by %s'%shifted_nnu_max)
    
    return np.abs(shifted_nnu_max - nnu_fid)

##################################################################################################################################