# duPont pipeline
Pipeline to detrend and calibrate observations with DirectCCD instrument at du Pont (DP) telescope for PHANGS

### PRE-PROCESSING ###
### (1) Put all raw DP data in a folder following the structure 
_{path_to_DP_data}/{night: utYYYYMMDD_DD}_
where with _DD_DD_ we indicate the beginning and ending day of the observation.
### (2) Check with the _list_in_browser.py_ script if the galaxy names and filter correspond
```sh
python list_in_browser.py -p {path_to_DP_raw_data}
```
### (3) Change galaxy name or filter in the raw data headers
```sh
python pre-processing.py -p {path_to_file_paths} --old_objname {wrong_galname} --new_objname {correct_name} --old_filtnames {filt1} {filt2} {filt3} --new_filtnames r Ha657 Ha663
```
The file where to store the paths to the raw data to update is _DirCCDData_paths.txt_.

### DATA REDUCTION ###

### Execute python scripts in the following order:
```sh
python dirCCDreduction.py {path_to_data_folder} {galaxy} {filter} -d --plot
```
where _{path_to_data_folder}_ is the one with the structure _utYYYYMMDD_DD_ and where the raw data are stored.
Note: the raw data are then moved to a folder named _input_.
```sh
python make_objects_txt_DP.py {path_to_data_folder} {galaxy}
```
The script will look either in the root where the raw data where originaly stored or in the _output/detrend_ where the detrended file are created after having run _dirCCDreduction.py_. In both ways DS9 is opened to save the centre of galaxy pixedl position for each frame in the same way as for WFI data reduction.
```sh
python dirCCDastrometry.py {path_to_data_folder} {galaxy} --plot --rotate_image
```
Move to the root pipeline folder to run the following scripts working on both WFI and DirectCCD observations: 
```sh
python BB_phot_calib.py --ccd_path {path_to_reproj} -w NGC0000 -c {galaxy} 
```
_{path_to_reproj}_ is the path to the galaxies reprojected onto the WCS solution. It is created in the main data root.
```sh
python NB_phot_calib.py --ccd_path {path_to_reproj} -w NGC0000 -c {galaxy}
```
```sh
python PSF_compute_gauss_to_gauss.py --ccd_path {path_to_calib} -w NGC0000 -c {galaxy} --bkg_subtraction
# (only the background subtraction part as indicated in the tutorial)
```
where _{path_to_calib} is the output folder wheere the photometrically calibrated galaxies are saved.
```sh
python weight_combine_v23.py --ccd_path {path_to_data_folder} --wfi_galaxies NGC0000 --ccd_galaxies {galaxy}
```

### POST-PROCESSING SCRIPTS FOR ASTROMETRIC REGISTRATION ###
This set of two scripts allows a further astrometric registration towards zero offset with Gaia DR2 stars matched with the stars in the DP images.
Execute the scripts in the following order:
```sh
python astrometry_tables_DP.py --path {path_to_final_images} --tbl_path {path_to_pipeline_folder}/CCD_tables --phot_sel --ccd_galaxies {galaxy} --gaia_dr GAIA-DR2
```
to write a table of Gaia DR2 stars in the galaxy field by reading the final ***comb*** images.
```sh
python astrometry_registration_plot_DP.py --path {path_to_final_images} --tbl_path {path_to_pipeline_folder}/CCD_tables --plot_path {path_to_pipeline_folder}/CCD_astrometry_plots --phot_sel --ccd_galaxies {galaxy} --gaia_dr GAIA-DR2 --correct_astrometry
```
to plot the RA-DEC plots of the offset positions between Gaia DR2 stars and the stars in DP images and to apply the mean offset to re-register the astrometry of the final images.
