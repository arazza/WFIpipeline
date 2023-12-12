# WFI pipeline
Pipeline to detrend and calibrate observations with WFI instrument at MPG 2.2-m telescope for PHANGS

### SOME VERY BASIC STEP-BY-STEP INSTRUCTIONS ###
### (1) Put all raw WFI data in a folder following the structure 
_{path_to_WFI_data}/{night: YYYY-MM-DD}/rawdata_
### (2) Uncompress fits.Z files with 
```sh
uncompress *fits.Z
```
### (3) Check files with
```sh
fitsheader *.fits -f -e 0 -k OBJECT -k "EXPTIME" -k "HIERARCH ESO DPR TYPE" -k "HIERARCH ESO INS FILT1 NAME"
```

### (4) Open one of the raw files and check in which CCD the centre of the galaxy is located and give the number to ***make_objects_txt_WFI.py*** with the optional keyword -c {number}
### (5) Execute ***make_objects_txt_WFI.py*** locally or with X forwarding, because a ds9 instance is needed
##### (5b) If this doesn't work well, you can also manually create/edit the output files ***{galaxy}\_Rc_{night}.txt*** and ***{galaxy}\_Ha_{night}.txt***. This file shows the RA/DEC and pixel coordinates of the galaxy centre, but a reference star can also be used.
##### (5c) The new observations seem to have "_large" and "_small" added to the actual object name in the header keyword. There is a function in ***make_objects_txt_WFI.py*** which removes these suffixes, whith will otherwise make the pipeline break. Call it if necessary with the --chkeys option.
### (6) Check if it is necessary to exclude CCDs from the reduction and add the numbers in WFIpipeline.config
### (7) Add galaxy to list in ***PHANGS_WFI_galaxies.txt***
### (8) Add path to ***paths_to_uncalib_WFI.txt*** in the form 
_{path_to_WFI_data}/{night YYYY-MM-DD}/processed/output/stacks_ 
### and to  paths_to_calib_WFI.txt in the form  
_{path_to_WFI_data}/{night YYYY-MM-DD}/processed/output/stacks/phot_calib_
### (9) Execute python scripts in the following order:

```sh
python make_objects_txt_WFI.py {path_to_WFI_data}/{night: YYYY-MM-DD}/rawdata {galaxy} phangs_sample_table_v1p6.fits -c {CCD number}
```
```sh
python CCDred_basic.py {path_to_WFI_data} {night} -o {galaxy} -jmbed
```
```sh
python WFIpipeline.py {path_to_WFI_data} {night} -o {galaxy} -a -c
```
```sh
python newmask_reprojection.py -g {galaxy}
```
```sh
python mask_saturated_pixels.py -g {galaxy}
```
```sh
python BB_phot_calib.py -w {galaxy} -c NGC0000 
```
```sh
python NB_phot_calib.py -w {galaxy} -c NGC0000
```
```sh
python make_galaxy_cutouts_v21.py -g {galaxy}
```
```sh
python PSF_compute_gauss_to_gauss.py -w {galaxy} -c NGC0000 --bkg_subtraction
# (only the background subtraction part as indicated in the tutorial)
```
```sh
python weight_combine_v23.py --wfi_galaxies {galaxy} --ccd_galaxies NGC0000
```

### POST-PROCESSING SCRIPTS FOR ASTROMETRIC REGISTRATION ###
This set of four scripts allows a further astrometric registration towards zero offset with Gaia DR2 stars matched with the stars in the WFI images.
Execute the scripts in the following order:
```sh
python Bkg_subtraction_full_FOV.py --wfi_fullFOV_path {path_to_WFI_calib} -w {galaxy}
```
to create the background subtracted images of each calibrated frame. An argument ***--frame_number {number_of_frame}*** can be added to start the computation from one specific frame in case of failure for running out of memory. 
```sh
python weight_combine_v23.py --wfi_path {path_to_WFI_calib} --wfi_galaxies {galaxy} --full_FOV
```
to combine the background-subtracted full FOV images.
```sh
python astrometry_tables_WFI.py --path {path_to_final_full_FOV} --tbl_path {path_to_pipeline_folder}/WFI_tables --phot_sel --wfi_galaxies {galaxy} --gaia_dr GAIA-DR2
```
to write a table of Gaia DR2 stars in the galaxy field by reading the final ***comb*** images.
```sh
python astrometry_registration_plot_WFI.py --path {path_to_final_full_FOV} --tbl_path {path_to_pipeline_folder}/WFI_tables --plot_path {path_to_pipeline_folder}/WFI_astrometry_plots --phot_sel --wfi_galaxies {galaxy} --gaia_dr GAIA-DR2 --correct_astrometry
```
to plot the RA-DEC plots of the offset positions between Gaia DR2 stars and the stars in WFI images ans to apply the mean offset to re-register the astrometry of the full FOV images
```sh
python astrometry_registration_plot_WFI.py --path {path_to_final_cutouts} --tbl_path {path_to_pipeline_folder}/WFI_tables --plot_path {path_to_pipeline_folder}/WFI_astrometry_plots --phot_sel --wfi_galaxies {galaxy} --gaia_dr GAIA-DR2 --correct_astrometry --cutouts
```
to do the same as above but to register the same astrometric solution found from the ***comb*** images into the cutouts.
