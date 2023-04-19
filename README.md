# WFIpipeline
Pipeline to detrend and calibrate observations with WFI instrument at MPG 2.2-m telescope for PHANGS

### SOME VERY BASIC STEP-BY-STEP INSTRUCTIONS ###
# (1) Put all raw WFI data in a folder following the structure {path_to_WFI_data}/{night: YYYY-MM-DD}/rawdata
# (2) Uncompress fits.Z files with uncompress *fits.Z
# (3) Check files with
> fitsheader *.fits -f -e 0 -k OBJECT -k "EXPTIME" -k "HIERARCH ESO DPR TYPE" -k "HIERARCH ESO INS FILT1 NAME"
# (4) Open one of the raw files and check in which CCD the centre of the galaxy is located and give the number to make_objects_txt_WFI.py with the optional keyword -c {number}
# (5) Execute make_objects_txt_WFI.py locally or with X forwarding, because a ds9 instance is needed
$ (5b) If this doesn't work well, you can also manually create/edit the output files {galaxy}_Rc_{night}.txt and {galaxy}_Ha_{night}.txt. This file shows the RA/DEC and pixel coordinates of the galaxy centre, but a reference star can also be used.
# (6) Check if it is necessary to exclude CCDs from the reduction and add the numbers in WFIpipeline.config
# (7) Add galaxy to list wfi_galaxies in BB_phot_calib.py and NB_phot_calib.py
# (8) Add path to paths_to_uncalib_WFI.txt in the form  {path_to_WFI_data}/{night YYYY-MM-DD}/processed/output/stacks and to  paths_to_calib_WFI.txt in the form  {path_to_WFI_data}/{night YYYY-MM-DD}/processed/output/stacks/phot_calib
# (9) Execute python scripts in the following order:

> python make_objects_txt_WFI.py {path_to_WFI_data}/{night: YYYY-MM-DD}/rawdata {galaxy} phangs_sample_table_v1p6.fits -c {CCD number}

> python CCDred_basic.py {path_to_WFI_data}  {night} -o {galaxy} -jmbed

> python WFIpipeline.py {path_to_WFI_data}  {night} -o {galaxy} -a -c

> python newmask_reprojection.py      # creates masks for all galaxies in paths_to_uncalib.txt - we should add a keyword to give a specific galaxy

> python mask_saturated_pixels.py     # creates masks for all galaxies in paths_to_uncalib.txt - we should add a keyword to give a specific galaxy

> python BB_phot_calib.py -w {galaxy} -c NGC0000 

> python NB_phot_calib.py -w {galaxy} -c NGC0000

> python make_galaxy_cutouts_v21.py 

> python PSF_compute_gauss_to_gauss.py -w {galaxy} -c NGC0000 --bkg_subtraction
# (only the background subtraction part as indicated in the tutorial)

> python weight_combine_v23.py --wfi_galaxies {galaxy} --ccd_galaxies NGC0000
