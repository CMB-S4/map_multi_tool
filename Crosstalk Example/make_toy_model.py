import numpy as np

#Making a toy model for 4x4 dichroic pixel wafer
pixel_size = 5300. #micro m
f_ratio = 1.9
aperture = 5.*(10**3) #mm

arcsec_per_pixel = (pixel_size/(f_ratio*aperture))*206.3
deg_per_pixel = arcsec_per_pixel/3600.
deg_per_microm = deg_per_pixel/pixel_size
        

pixelRow = [0,0,0,0,
            0,0,0,0,
            0,0,0,0,
            0,0,0,0,
            
            1,1,1,1,
            1,1,1,1,
            1,1,1,1,
            1,1,1,1,
            
            2,2,2,2,
            2,2,2,2,
            2,2,2,2,
            2,2,2,2,
            
            3,3,3,3,
            3,3,3,3,
            3,3,3,3,
            3,3,3,3]

pixelCol = [0,0,0,0,
            1,1,1,1,
            2,2,2,2,
            3,3,3,3,
           
            0,0,0,0,
            1,1,1,1,
            2,2,2,2,
            3,3,3,3,
            
            0,0,0,0,
            1,1,1,1,
            2,2,2,2,
            3,3,3,3,
            
            0,0,0,0,
            1,1,1,1,
            2,2,2,2,
            3,3,3,3,]

pixelXpos = [-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
             (pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),
            
            -(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
             (pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),
            
            -(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
             (pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),
            
           -(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
            (pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),

] #micro m

pixelYpos = [(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),
             (pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),
             (pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),
             (pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),(pixel_size)+1*(pixel_size/2.),

             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
             1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),1*(pixel_size/2.),
            
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
             -1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),-1*(pixel_size/2.),
            
             -(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),
             -(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),
             -(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),
             -(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.),-(pixel_size)-1*(pixel_size/2.)] #micro m

detector = ['L00x','L00y','H00x','H00y',
           'L01a','L01b','H01a','H01b',
           'L02x','L02y','H02x','H02y',
           'L03a','L03b','H03a','H03b',
            
           'L10a','L10b','H10a','H10b',
           'L11x','L11y','H11x','H11y',
           'L12a','L12b','H12a','H12b',
           'L13x','L13y','H13x','H13y',
            
           'L20x','L20y','H20x','H20y',
           'L21a','L21b','H21a','H21b',
           'L22x','L22y','H22x','H22y',
           'L23a','L23b','H23a','H23b',
            
           'L30a','L30b','H30a','H30b',
           'L31x','L31y','H31x','H31y',
           'L32a','L32b','H32a','H32b',
           'L33x','L33y','H33x','H33y']

angle = [0.0,90.0,0.0,90.0,
         45.0,135.0,45.0,135.0,
         0.0,90.0,0.0,90.0,
         45.0,135.0,45.0,135.0,
         
         45.0,135.0,45.0,135.0,
         0.0,90.0,0.0,90.0,
         45.0,135.0,45.0,135.0,
         0.0,90.0,0.0,90.0,
        
         0.0,90.0,0.0,90.0,
         45.0,135.0,45.0,135.0,
         0.0,90.0,0.0,90.0,
         45.0,135.0,45.0,135.0,
        
         45.0,135.0,45.0,135.0,
         0.0,90.0,0.0,90.0,
         45.0,135.0,45.0,135.0,
         0.0,90.0,0.0,90.0]

skyX = [] 
skyY = []

i=0 
while i<len(pixelXpos):
    skyX.append(pixelXpos[i]*deg_per_microm)
    skyY.append(pixelYpos[i]*deg_per_microm)
    i+=1
    
#Creating data file for toy model wafer to feed into MMT
with open('/Users/cesileyking/Desktop/GitHub/map_multi_tool/data_4x4.txt','w') as f:
    i=0
    while i<len(detector):
        f.write(str(detector[i])+' '+str(skyX[i])+' '+str(skyY[i])+' '+str(angle[i])+'\n')
        i+=1