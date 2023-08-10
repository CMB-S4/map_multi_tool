# map_multi_tool

The Map Multi Tool is a collection of python packages that allows the user to model some systematics in the map domain. There are currently three packages:
* MMTModules.py - The main package that contains the modules to import focal plane definitions, calculate crosstalk, and plot 'beam maps'
* DetectorTimeConstantModules.py - The package specifically made to model the map effects of imperfectly measured detector time constants. This module contains functions to define distributions of detector time constants and measurements, smear beam maps along a scan direction, rotate that scan direction.
* CobayaModules.py - Contains functions to fit power spectra to a fiducial LCDM model using a Cobaya minimizer.

The math behind the map multi tool can be summarized by matrix mapping equation:

$$\begin{pmatrix}
I_{detect} \\ 
Q_{detect} \\ 
U_{detect}\end{pmatrix} = \begin{pmatrix}
II & IQ & IU \\
QI & QQ & QU \\
UI & UQ & UU \end{pmatrix} 
\begin{pmatrix}I_{sky} \\
Q_{sky} \\
U_{sky}\end{pmatrix}$$

where the components of the 3x3 beam coupling matrix are calculated by MMT.

# Getting Started

Once this repo is cloned, import 'MMTModules.py' into whatever python analysis environment you are using (also import whatever other packages you think may be useful). If you want to analyze a real focal plane, then you will need a text or csv file that contains detector properties: det ID, xy positions (in degrees on the sky), pol angle (in degrees). The detector ID should be in the format waferNumber_detectorIndex_bandID_polAngleOrientation. An example is given in "wafer_99.txt" with the first entry being "99_000_MFL1_A". The function generate_focal_plane_distribution() takes this file and puts the information in a dictionary for the rest of the software to read. 

For a crosstalk analysis, the user must also feed information about the detector couplings to generate a crosstalk matrix. This can be done randomly from the function generate_random_couplings() or for a rhombus layout generate_bondpad_couplings_rhombus(). Both of these functions take the detector dictionary from the first paragraph as an input. The user could also define their own coupling mechanism. 

There are many other inputs that deal with the size of the beam, scan speed, frequency bands, etc are detailed in the jupyter notebook 'MMTModuels_Example.ipynb' in this repo. This notebook goes thru a crosstalk analysis using an arbitrary set of observation and map making parameters. 

If you are not sure what a function does each one is docstringed!

# Assumptions

The map multi tool generates composite focal plane maps in IQU space from the observation of a point source at the center of the map. That being said, there are many assumptions this modeling makes detailed below (and possibly more):

* Zero gain mismatch between detectors (no calibration differences)
  * This could be added later by a motivated user
* Each detector has unit response to the point source (flat and equal frequency bands)
* All crosstalk at the same percent level
* Gaussian instrument beam
* Listed examples assume a purely temperature sky
* Gaussian Likelihoods (for Cobaya fitting)
