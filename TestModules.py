#theta in degrees
def calc_quadrupole(r, theta, p, rotation_angle=0):
    import numpy as np
    #ep_0 = 8.857187*10**(-12)
    return 3*p*np.cos(theta-rotation_angle)*np.sin(theta-rotation_angle)/(r**3.)

def calc_dipole_onaxis(r, theta, p):
    import numpy as np
    #ep_0 = 8.857187*10**(-12)
    return p*(3*np.cos(theta)**2. - 1)/(r**3.)

def calc_dipole(r, theta, p, rotation_angle=0):
    import numpy as np
    
    
    return 2*p*np.sin(theta-rotation_angle) / r**3.

def dipole_field(beam_map, p, rotation_angle=0):
    import numpy as np
    
    #square map
    N = len(beam_map)
    
    #coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.)
    X = np.outer(ones,inds)
    Y = -np.transpose(X)
    
    theta_map = np.arctan2(Y,X)
    for pix_col in range(len(beam_map)):
        for pix_row in range(len(beam_map[pix_col,:])):
            theta = theta_map[pix_col,pix_row]
            r = np.sqrt( (pix_col - int(N/2))**2. + (pix_row - int(N/2))**2. )
#            try:
#                theta = np.arctan((pix_col- int(N/2)) / (pix_row-int(N/2)))
#            except ZeroDivisionError:
#                if pix_col < 512:
#                    theta = np.pi/2
#                else:
#                    theta = 3*np.pi/2
            if r == 0:
                dipole_component = 1
            else:
                #dipole_component = calc_dipole_onaxis(r,theta,p)
                dipole_component = calc_dipole(r,theta,p,rotation_angle=rotation_angle)
                
                
            if dipole_component > 1:
                beam_map[pix_col,pix_row] = 1
            else:
                beam_map[pix_col,pix_row] = dipole_component
            
    
    return beam_map

def quadrupole_field(beam_map, Q, rotation_angle=0):
    import numpy as np
    
    #square map
    N = len(beam_map)
    #coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.)
    X = np.outer(ones,inds)
    Y = -np.transpose(X)
    
    theta_map = np.arctan2(Y,X)
    for pix_col in range(len(beam_map)):
        for pix_row in range(len(beam_map[pix_col,:])):
            theta = theta_map[pix_col,pix_row]
            r = np.sqrt( (pix_col - int(N/2))**2. + (pix_row - int(N/2))**2. )
#            try:
#                theta = np.arctan( (pix_col-int(N/2)) / (pix_row-int(N/2)) )
#            except ZeroDivisionError:
#                if pix_col < 512:
#                    theta = np.pi/2
#                else:
#                    theta = 3*np.pi/2
            
            if r == 0:
                quadrupole_component = 0
            else:
                quadrupole_component = calc_quadrupole(r,theta,Q,rotation_angle=rotation_angle)
                
            if quadrupole_component > 1:
                beam_map[pix_col,pix_row] = 1
            else:
                beam_map[pix_col,pix_row] = quadrupole_component
    
    return beam_map