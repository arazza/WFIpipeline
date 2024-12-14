# -*- coding: utf-8 -*-
"""
Created on Fri May 3 17:00 2019

@author: arazza
"""

import re
import os
import timeit
import argparse
import sys
import warnings

import numpy as np
from numpy.linalg import lstsq

from pathlib import Path

from astropy.io import fits
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.nddata import NDData
from astropy.visualization import ZScaleInterval, simple_norm
from astropy.utils.data import conf

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astroquery.vizier import Vizier

from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.aperture import CircularAperture, SkyCircularAperture
from photutils.psf import extract_stars, EPSFBuilder

from ccdproc import ImageFileCollection

from dirCCDreduction import ImagesPaths




class AstrometricImages (object) :

    def __init__ (self, paths, files) :

        self.paths = paths
        self.files = files
        # Get the object name
        name = os.path.basename(self.files[0])
        pos = name.find('_')   
        self.obj = name[:pos]
        # satellite tracks files
        self.st_files = np.loadtxt('satellite_tracks.txt', dtype='str', usecols=(0,))


    def init_paths (self) :
        
        #self.paths.out_catalogs_dir = self.paths.outdir.joinpath('catalogs')
        #self.paths.out_wcs_dir = self.paths.outdir.joinpath('wcs_corrected')
        #self.paths.out_mask_dir = self.paths.outdir.joinpath('mask')
        #self.paths.out_unc_dir = self.paths.outdir.joinpath('uncertainty')
        #self.paths.out_reproj_dir = self.paths.outdir.joinpath('reproj')
        self.paths.catalogs_dir = self.paths.path.parent.joinpath('catalogs')
        self.paths.wcs_dir = self.paths.path.parent.joinpath('wcs_corrected')
        self.paths.mask_dir = self.paths.path.parent.joinpath('mask')
        self.paths.unc_dir = self.paths.path.parent.joinpath('uncertainty')
        self.paths.reproj_dir = self.paths.path.parent.joinpath('reproj')

        #self.paths.out_catalogs_dir.mkdir(parents=True, exist_ok=True)
        #self.paths.out_wcs_dir.mkdir(parents=True, exist_ok=True) 
        #self.paths.out_mask_dir.mkdir(parents=True, exist_ok=True)
        #self.paths.out_unc_dir.mkdir(parents=True, exist_ok=True)
        #self.paths.out_reproj_dir.mkdir(parents=True, exist_ok=True)
        self.paths.catalogs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.wcs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.mask_dir.mkdir(parents=True, exist_ok=True)
        self.paths.unc_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reproj_dir.mkdir(parents=True, exist_ok=True)

        self.checkplots_path = Path('.').joinpath('checkplots')
        self.checkplots_path.mkdir(parents=True, exist_ok=True)


    def rough_wcs (self, rotate, from_name) :

        def compute_line (x1, y1, x2, y2) :

            points = [(x1, y1),(x2, y2)] 
            x_coords, y_coords = zip(*points)
            A = np.vstack([x_coords,np.ones(len(x_coords))]).T 

            return lstsq(A, y_coords)[0] 

        # Define again detrend dir that is instantaited only during detrend
        self.paths.detrend_dir = self.paths.outdir.joinpath('detrend')

        for filt in ['r', 'Ha657', 'Ha663'] :
            
            obj_txt = '%s_%s_%s.txt' %(self.obj, filt, self.paths.path.stem)
            path_to_obj_txt = Path('.').joinpath('objects_txt', obj_txt)

            if path_to_obj_txt.exists() :
                
                x, y = np.loadtxt (str(path_to_obj_txt), dtype='str', unpack=True)
                
                if from_name :
                    self.obj_coord = SkyCoord.from_name(self.obj)
                else :
                    self.obj_coord = SkyCoord (ra=x[0], dec=y[0], unit=(u.hourangle, u.deg))
                
                crpix1 = x[1:].astype(float)
                crpix2 = y[1:].astype(float)
                
                list_of_files = sorted(self.paths.detrend_dir.glob('%s_%s*'%(self.obj,filt)))
                
                list_of_files = self.sort_nicely([str(file) for file in list_of_files])
                 
                list_of_files = [Path(file) for file in list_of_files]
                                
                for i,file in enumerate(list_of_files) :
                    
                    ### Check if the mask needs to be updated for the satellite tracks
                    if file.name in self.st_files :
                            
                        with open('satellite_tracks.txt') as f :
                                
                            for line in f.readlines() :
                                cols = line.split()
                                    
                                if cols[0] == file.name :
                                    
                                    x1 = int(cols[1])   
                                    y1 = int(cols[2])
                                    x2 = int(cols[3])
                                    y2 = int(cols[4])
                                    
                                    m1, c1 = compute_line(x1, y1, x2, y2)
                                    print(m1, c1)
                                    
                                    x3 = int(cols[5])   
                                    y3 = int(cols[6])
                                    x4 = int(cols[7])
                                    y4 = int(cols[8])
                                    
                                    m2, c2 = compute_line(x3, y3, x4, y4)
                                    print(m2, c2)
                                    
                                    # Open the mask extension
                                    hdulist = fits.open(file)
                                    mask = hdulist['MASK'].data
                                    
                                    # Update the mask
                                    print ('\nUpdating %s\nto mask the satellite track ...'%file)
                                    xmin = min(x1,x2,x3,x4)
                                    xmax = max(x1,x2,x3,x4)
                                    ymin = min(y1,y2,y3,y4) 
                                    ymax = max(y1,y2,y3,y4)
                                    for y in range(ymin,ymax) : 
                                        for x in range(xmin,xmax) : 
                                            if (y>=m1*x+c1) and (y<=m2*x+c2) : 
                                                mask[y,x]=1  
                                    
                                    #fig, ax = plt.subplots(1,1, figsize=(7,7))
                                    #ax.imshow(mask, origin='lower')
                                    hdulist['MASK'].data = mask
                                    hdulist.writeto(file, overwrite=True)
                                    
                    ### Updating the header with the rough wcs                            

                    print ('\nUpdating the header of\n%s\nwith a rough WCS solution ...'%file) 

                    hdulist = fits.open(str(file))
                    hdr = hdulist[0].header
                    
                    # write the coordinates of the reference point (centre of the galaxy, standard star)
                    hdr.set('CRVAL1', self.obj_coord.ra.value, '[deg] Coordinate value at reference point')
                    hdr.set('CRVAL2', self.obj_coord.dec.value, '[deg] Coordinate value at reference point')

                    pixscale = hdr['SCALE'] / 3600.

                    '''
                    Apply the transformation Theta --> 90, X --> -X 
                    Ra  = -pixscale * (X - CRPIX1)
                    Dec = -pixscale * (Y - CRPIX2)
                    '''
                    hdr.set('CRPIX1', crpix1[i], 'Pixel coordinate of reference point') 
                    hdr.set('CRPIX2', crpix2[i], 'Pixel coordinate of reference point') 
                    if rotate :
                        hdr.set('CD1_1', 0, 'Linear projection matrix')
                        hdr.set('CD2_2', 0, 'Linear projection matrix')
                        hdr.set('CD1_2', -pixscale, 'Linear projection matrix')
                        hdr.set('CD2_1', -pixscale, 'Linear projection matrix')
                    else :
                        hdr.set('CD1_1', pixscale, 'Linear projection matrix')
                        hdr.set('CD2_2', -pixscale, 'Linear projection matrix')
                        hdr.set('CD1_2', 0, 'Linear projection matrix')
                        hdr.set('CD2_1', 0, 'Linear projection matrix')
                    hdr.set('CTYPE1', 'RA---TAN', 'Right ascension, gnomonic projection')
                    hdr.set('CTYPE2', 'DEC--TAN', 'Declination, gnomonic projection')
                    hdr.set('CUNIT1', 'deg', 'Units of coordinate increment and value')
                    hdr.set('CUNIT2', 'deg', 'Units of coordinate increment and value')

                    hdulist[0].header = hdr

                    hdulist.writeto(file, overwrite=True)          

            else :

                continue


    def build_cmd(self, filenames, **kwargs):
        """
        Build a command to run an astromatic code.
        
        Parameters
        ----------
        filenames: str or list
            Name of a file or list of filenames to run in the command line statement
        **kwargs: keyword arguments
            The following are optional keyword arguments that may be used:
                - code: str
                    Name of the astromatic code to use. This should be contained in 
                    ``astrotoyz.astromatic.api.codes``
                - config: dict (optional)
                    Dictionary of configuration options to pass in the command line.
                - config_file: str (optional)
                    Name of the configuration file to use. If none is specified, the default
                    config file for the given code is used
        
        Returns
        -------
        cmd: str
            Commandline statement to run the given code
        kwargs: dict
            Dictionary of keyword arguments used in the build
        """
        
        # If a single catalog is passed, convert to an array
        if not isinstance(filenames, list):
            filenames = [filenames]
        
        # If the user did not specify a params file, create one in the temp directory and 
        # update the config parameters
        if kwargs['cmd']=='sex':
           
            param_name = Path('.').joinpath('sex.param')
            
            with param_name.open('w') as f :
                for p in kwargs['params']:
                    f.write(p+'\n')
            
            kwargs['config']['PARAMETERS_NAME'] = str(param_name)

        # Get the correct command for the given code (if one is not specified)
        cmd = kwargs['cmd']
        if cmd[-1]!=' ':
            cmd += ' '
        # Append the filename(s) that are run by the code
        cmd += ' '.join(filenames)
        
        # Add on any user specified parameters
        for param in kwargs['config']:

            if isinstance(kwargs['config'][param], bool):
                if kwargs['config'][param]:
                    val='Y'
                else:
                    val='N'
            else:
                val = kwargs['config'][param]
            cmd += ' -'+param+' '+val
        
        #return (cmd, kwargs)
        return cmd
    

    def clean_files (self) :
        '''
        Clean the ldac, newldac, head and wcs files if you have to overwrite the lists
        '''
        
        # Make lists and path to catalogs directoies
        cat_path_list = [path for path in sorted(self.paths.catalogs_dir.glob('%s*'%self.obj))]
        wcs_path_list = [path for path in sorted(self.paths.wcs_dir.glob('%s*'%self.obj))]
        mask_path_list = [path for path in sorted(self.paths.mask_dir.glob('%s*'%self.obj))]
        unc_path_list = [path for path in sorted(self.paths.unc_dir.glob('%s*'%self.obj))]
        reproj_path_list = [path for path in sorted(self.paths.reproj_dir.glob('%s*'%self.obj))]
        pathtotxt = self.paths.catalogs_dir.joinpath('%s_detrended_files.txt'%self.obj)

        # Clean the files folder if new files has to overwrite the old ones
        if cat_path_list!=[] :
            # Remove the existing catalog files
            for p in cat_path_list :
                p.unlink()
        if wcs_path_list!=[] :
            # Remove the existing wcs-corrected files
            for p in wcs_path_list :
                p.unlink()
        if mask_path_list!=[] :
            # Remove the existing mask files
            for p in mask_path_list :
                p.unlink()
        if unc_path_list!=[] :
            # Remove the existing uncertainty files
            for p in unc_path_list :
                p.unlink()
        if reproj_path_list!=[] :
            # Remove the existing reprojected files
            for p in reproj_path_list :
                p.unlink()
        # Remove the existing text file where paths to catalog files are linked to paths to detrended files
        if pathtotxt.exists() :
            pathtotxt.unlink()

     
    def sort_nicely (self, l): 
        ''' 
        Sort the given list in the way that humans expect (human sorting). 
        Natural (human) sort fixes the problem of frame #10 being the last 
        created but #9 the last one in the list generated with glob method. 
        Withouth this fix #10 is overwritten N times by N new frames. 
        ''' 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
        
        return sorted(l, key=alphanum_key)
        
        
    def run_sextractor (self) :
        
        print ('\nRunning SExtractor')
        print('\t for data in %s ...'%self.paths.detrend_dir)
            
        self.cat_dir = self.paths.catalogs_dir
        self.pathtotxt = self.paths.catalogs_dir.joinpath('%s_detrended_files.txt'%self.obj)
        
        # Create the list of frames with their paths and define SExtractor parameters
        
        self.catalog_names = []
        
        for file in self.files :

            pathtofile = self.paths.detrend_dir.joinpath(file)
            
            # Create a human ordered list of the calalogs files already present in catalog directory
            pos = len(file)-file[::-1].find('_')    # find position of the second '_' within the file name
            cat_list = [str(path) for path in sorted(self.cat_dir.glob('%s*_ldac.fits'%file[:pos]))]
            cat_sorted_list = self.sort_nicely(cat_list)
            
            # assign an incremental number if there are already files in catalog directory
            if cat_sorted_list!=[] :
                last_fname = Path(cat_sorted_list[-1])
                frame_index = int(last_fname.stem[pos:-5])
                file = file[:pos] + str(frame_index + 1) + file[-5:]

            pathtocat = self.cat_dir.joinpath(file.replace('.fits','_ldac.fits'))
            
            self.catalog_names.append(str(pathtocat))

            # Append the detrended file together with its related catalog file with their full paths in a text file
            with self.pathtotxt.open(mode='a') as f :
                f.write('%s %s\n'%(pathtofile,pathtocat)) 

            kwargs = {
            # Command to run SExtractor
            'cmd': 'sex',
            # Paramteres in the configuration file
            'config': {
                'CATALOG_NAME': self.catalog_names[-1],
                'CATALOG_TYPE': 'FITS_LDAC',
                'FILTER': 'N'#,
                #'SATUR_LEVEL': '500',
                #'GAIN' : '1.0'
            },
            # Output parameters
            'params': ['NUMBER', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE', 'ERRTHETAWIN_IMAGE',     # for SCAMP mandatory
                        'FLUX_AUTO', 'FLUXERR_AUTO','FLUX_RADIUS', 'FLAGS', 
                        'FLUX_PETRO', 'FLUXERR_PETRO', 'FWHM_IMAGE', 'FWHMPSF_IMAGE', 'CLASS_STAR']
            }

            # Build the command to run SExtractor
            self.cmd_sex = self.build_cmd(str(pathtofile), **kwargs)

            # Run SExtractor
            os.system (self.cmd_sex)
    

    def filter_catalog (self, exclude) :

        for i, catalog in enumerate(self.catalog_names) :
        
            hdulist = fits.open(catalog)
            tbldata = hdulist['ldac_objects'].data
            mask = (tbldata['FLUX_AUTO']<=7000)*(tbldata['FLUX_AUTO']>=20)*(tbldata['FLAGS']==0)*(tbldata['FLUX_RADIUS']>=0)
            
            newtbldata = tbldata[mask]
            tblhdu = fits.BinTableHDU (data=newtbldata, name='LDAC_OBJECTS')
            newhdulist = fits.HDUList ([hdulist['primary'], hdulist['ldac_imhead'], tblhdu])
            
            newcatalog = catalog.replace('_ldac.fits','_newldac.fits')
            newhdulist.writeto(newcatalog, overwrite=True)
            self.catalog_names[i] = newcatalog
        
        # Get the whole list of catalogs
        self.catalog_names = [str(path) for path in sorted(self.cat_dir.glob('%s*_newldac.fits'%self.obj))]  
        
        '''
        Exclude files to compute the astrometry. Important in case some of the frames degrade the solution.
        NOTE: files to be excluded has to be given with the full path in case you are excluding files from a
        folder which is not the current one. 
        Important: self.catalog_names has to be transformed to an numpy.array to handle the numpy.where function
        and then transformed back to a list to be used by the following methods.
        '''
        self.pathtofile, self.pathtocat =  np.loadtxt(self.pathtotxt, dtype='str', unpack=True)
        
        self.catalog_names = np.array(self.catalog_names)

        for file in exclude :
            headfile = self.cat_dir.joinpath(Path(file).stem + '_newldac.head')  
            if headfile.exists() :
                headfile.unlink() 
            if Path(file).is_absolute() :
                index = np.where(self.pathtofile == file)[0]
            else :    
                index = np.where(self.pathtofile == str(self.paths.detrend_dir.joinpath(file)))[0]
            if index.size == 0 :
                continue
            cat = self.pathtocat[index][0].replace('_ldac','_newldac')
            catindex = np.where(self.catalog_names==cat)[0]
            self.catalog_names = np.delete(self.catalog_names, catindex) 
        
        self.catalog_names = list(self.catalog_names)
    
    

    def run_scamp (self, ref_catalog='GAIA-DR1', ref_band='DEFAULT', dist_degree = '3', nthrds = 0) :
        
        # Get astrometric solution from SCAMP

        print ('\nRunning SCAMP')
        print('\t for data in %s ...'%self.paths.detrend_dir)
        
        mergedname = str(self.cat_dir.joinpath('%s_scamp.cat'%self.obj)) 
        fullname = str(self.cat_dir.joinpath('%s_full.cat'%self.obj))
        
        checkplot_list = ['fgroups','distort','astr_interror2d','astr_interror1d','astr_referror2d','astr_referror1d','astr_chi2','psphot_error']
        checkplot_pathlist = [str(self.cat_dir.joinpath(checkplot)) for checkplot in checkplot_list]
        checkplots = ','.join(checkplot_pathlist)
        
        kwargs = {
            'cmd': 'scamp',
            'config': {
                #'MATCH': 'N',
                #'MATCH_FLIPPED': 'Y',
                'ASTREF_CATALOG': ref_catalog,
                'ASTREF_BAND': ref_band,
                'ASTREFMAG_LIMITS': '13.0,19.0',    # magnitude limits based on ZP plots
                'DISTORT_DEGREES': dist_degree, 
                'SOLVE_PHOTOM': 'N',
                #'GAIN' : '1.0',
                #'CHECKPLOT_DEV': 'NULL'
                'SAVE_REFCATALOG': 'Y', 
                'REFOUT_CATPATH': str(self.paths.catalogs_dir),
                'MERGEDOUTCAT_TYPE': 'ASCII_HEAD',
                'MERGEDOUTCAT_NAME': mergedname,
                'FULLOUTCAT_TYPE': 'ASCII_HEAD',
                'FULLOUTCAT_NAME': fullname,
                'CHECKPLOT_NAME': checkplots,
                'NTHREADS': '%s'%nthrds
            }
            #'config_file': 'config.scamp'
        }
        
        # Build the command to run SCAMP
        self.cmd_scamp = self.build_cmd(self.catalog_names, **kwargs)
        
        # Run SCAMP
        os.system (self.cmd_scamp)
        

    def save_wcs (self) :

        # Create wcs-corrected image frames, with updated headers, mask and error frames

        self.wcs_filenames = []
        
        self.pathtofile = [Path(file) for file in self.pathtofile] 
        self.pathtocat = [Path(cat) for cat in self.pathtocat]

        for file, cat in zip(self.pathtofile, self.pathtocat) :

            ## Instantiate the path to file
            #pathtofile = self.paths.detrend_dir.joinpath(file)
            
            # Check the object type to decide where to save the output  
            wcsfile = self.paths.wcs_dir.joinpath(cat.stem.replace('_ldac','_wcs.fits'))
            maskfile = self.paths.mask_dir.joinpath(cat.stem.replace('_ldac','_mask.fits'))
            stdfile = self.paths.unc_dir.joinpath(cat.stem.replace('_ldac','_unc.fits'))

            # load image
            hdu = fits.open(file)
            data = hdu[0].data
            header = hdu[0].header
            # import scamp output .head file
            newcat_file = self.cat_dir.joinpath(cat.stem.replace('_ldac','_newldac.head'))
            # Check if the .head file was not created (excluded by the parsed argument --exclude)
            if newcat_file.exists() :
                # Append in the list of files to be used to find the common reprojection
                self.wcs_filenames.append(str(wcsfile))

                # Correct the 'é' character causing the fits.Header.fromtextfile to fail
                with open(newcat_file, 'r') as f :
                    
                    lines = f.readlines() 
                
                with open(newcat_file, 'w') as f :
                    
                    for line in lines :
                        
                        if line.strip("\n")=='COMMENT   (c) 2010-2018 Sorbonne Université/Université de Bordeaux/CNRS' :
                            f.write('COMMENT   (c) 2010-2018 Sorbonne Universite/Universite de Bordeaux/CNRS')
                        
                        else :
                            f.write(line)


                newcat_header = fits.Header.fromtextfile(str(newcat_file))
                
                # merge image header with scamp output header (updating the old header cards)
                header.extend (newcat_header.cards, update=True)
            
            # save wcs files updating the image header
            print ('Writing to ...\n %s' %wcsfile)
            fits.writeto (wcsfile, data, header=header, overwrite=True)
            
            # save mask frames
            maskfits = hdu['MASK']
            mask_temp = np.array(np.zeros(maskfits.data.shape), float)
            mask_temp[maskfits.data==1] = 1
            mask_temp[maskfits.data==0] = 99
            maskfits.data = mask_temp
            print ('Writing to ...\n %s' %maskfile)
            fits.writeto (maskfile, maskfits.data, header=header, overwrite=True)
                        
            # save variance map frames
            stdfits = hdu['UNCERT']
            print ('Writing to ...\n %s' %stdfile)
            fits.writeto (stdfile, stdfits.data, header=header, overwrite=True)


    def run_swarp (self, nthrds = 0) :
        '''
        Run Swarp on all the astrometrically calibrated observations, in all filters, belonging to a target.
        A place-folder is then created to be used to reporoject any single frame onto it.
        '''
       
        # Resample (rotate and scale) and combine (stack) images
        
        # Define the directories where wcs corrected images are stored and where the reprojected image has to be saved
        wcs_corr_dir = self.paths.wcs_dir
        reproj_dir = self.paths.reproj_dir

        print ('\nRunning SWARP')
        print('\t for data in %s ...'%wcs_corr_dir)
    
        self.reproj_filename = reproj_dir.joinpath('%s_reproj.fits'%self.obj)
        kwargs = {
            'cmd': 'swarp',
            'config': {
                'IMAGEOUT_NAME': str(self.reproj_filename),
                'SUBTRACT_BACK' :'N',
                'NTHREADS': '%s'%nthrds
            }
        }
        
        #self.wcs_filenames = [str(wcs_corr_dir.joinpath(cat.stem.replace('_ldac','_wcs.fits'))) for cat in self.pathtocat]

        # Build the command to run SWARP
        self.cmd_swarp = self.build_cmd(self.wcs_filenames, **kwargs)
        
        # Run SWARP
        os.system (self.cmd_swarp)



    def astr_precision (self, doplot) :
        '''
        This method estimates the astrometric precision of the common projection created for each target.
        A check plot is saved if requested by the user.
        '''
        
        # Instantiate the ProjImage class
        self.proj_img = ProjImage(self.reproj_filename)

        # Find stars with IrafStarFinder algorithm
        self.proj_img.find_image_stars()
        
        # Instantiate Gaia stars
        gs = GAIAStars(self.obj)
        gs.gaia_query()

        # Match stars found in the stacked image (common projection) with Gaia stars in the image field
        self.proj_img.match_with_gaia(gs)

        # Find a PSF model to re-fit the centroids
        self.proj_img.img_psf_fitting(doplot)
        
        # Estimate the astrometric precision
        print('\nCompute the astrometric precision and write it in the header ..\n')
        
        mytbl = self.proj_img.tbl
        
        ra_mean_sep = np.mean((mytbl['gaia_ra']-mytbl['ra_fit']).to(u.arcsec)) 
        dec_mean_sep = np.mean((mytbl['gaia_dec']-mytbl['dec_fit']).to(u.arcsec))
        ra_std_sep = np.std((mytbl['gaia_ra']-mytbl['ra_fit']).to(u.arcsec)) 
        dec_std_sep = np.std((mytbl['gaia_dec']-mytbl['dec_fit']).to(u.arcsec))

        gaia = SkyCoord(ra=mytbl['gaia_ra'], dec=mytbl['gaia_dec'], frame='icrs')
        img = SkyCoord(ra=mytbl['ra_fit'], dec=mytbl['dec_fit'], frame='icrs')
        seps = gaia.separation(img)
        mean_sep = np.mean(seps.to(u.arcsec))
        std_sep = np.std(seps.to(u.arcsec))

        # Save the astrometric precision in the reprojected image header
        print('\nSaving the following estimates of the astrometric solution in the header ...')
        print('Number of stars used to estimate the astrometric solution: ',len(mytbl))
        print('Mean separation in RA with Gaia DR2 [arcsec]: ',ra_mean_sep.value)
        print('Mean separation in DEC with Gaia DR2 [arcsec]: ',dec_mean_sep.value)
        print('Std of separation in RA with Gaia DR2 [arcsec]: ',ra_std_sep.value)
        print('Std of separation in DEC with Gaia DR2 [arcsec]: ',dec_std_sep.value)
        print('Mean angular separation with Gaia DR2 [arcsec]: ',mean_sep.value)
        print('Std of the angular separation with Gaia DR2 [arcsec]: ',std_sep.value)
        self.proj_img.hdr.set('NSTARS', len(mytbl), '# of stars to estimate the astrometric solution')
        self.proj_img.hdr.set('RASEP', ra_mean_sep.value, '[arcsec] Mean separation in RA (Gaia DR2)')
        self.proj_img.hdr.set('DESEP', dec_mean_sep.value, '[arcsec] Mean separation in DEC (Gaia DR2)')
        self.proj_img.hdr.set('RASEPRMS', ra_std_sep.value, '[arcsec] Std of separation in RA (Gaia DR2)')
        self.proj_img.hdr.set('DESEPRMS', dec_std_sep.value, '[arcsec] Std of separation in DEC (Gaia DR2)')
        self.proj_img.hdr.set('ANGSEP', mean_sep.value, '[arcsec] Mean ang separation (Gaia DR2)')
        self.proj_img.hdr.set('ANGRMS', std_sep.value, '[arcsec] Std of the ang separation (Gaia DR2)')

        hdulist = fits.open(str(self.proj_img.filename))
        hdulist[0].header = self.proj_img.hdr
        hdulist.writeto(str(self.proj_img.filename), overwrite=True)   

        # Check plot for the astrometry
        if doplot :
            # Define postions and apertures
            img_positions = SkyCoord(ra=mytbl['ra_fit'], dec=mytbl['dec_fit'])
            img_apertures = SkyCircularAperture(img_positions, r=8. * u.arcsec)
            gaia_positions = SkyCoord(ra=mytbl['gaia_ra'], dec=mytbl['gaia_dec'])
            gaia_apertures = SkyCircularAperture(gaia_positions, r=4. * u.arcsec)
            
            # Plot image and apertures for visual check
            fig, ax = plt.subplots(1, 1, figsize=(6.92, 7))
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(self.proj_img.data)
            ax.imshow(self.proj_img.data, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
            img_apertures.to_pixel(self.proj_img.wcs).plot(ax=ax, color='C0', lw=1.5, alpha=0.5)
            gaia_apertures.to_pixel(self.proj_img.wcs).plot(ax=ax, color='C1', lw=1.5, alpha=0.5)
            ax.set_title(self.proj_img.obj)
            #plt.show()
            fig.savefig(self.checkplots_path.joinpath(str(self.reproj_filename.name).replace('.fits','_match.png')))
            plt.close()   


    def reproj_single_frame (self, nthrds = 0) :
        '''
        Method to perform the reprojecton of the science, standard deviation and mask images onto a common place-folder
        Headers are updated by merging those after the detrend with the new ones after the reprojection to propagate Swarp solution.
        '''
        
        # Basic objects to use
        reproj_dir = self.reproj_filename.parent
        reproj_hdr = self.proj_img.hdr
        
        # Commoon projection parameters
        image_size = [reproj_hdr['NAXIS1'],reproj_hdr['NAXIS2']]
        image_centre = [reproj_hdr['CRVAL1'],reproj_hdr['CRVAL2']]
        pixel_scale = float(reproj_hdr['CD2_2'])*3600
        
        for file, cat in zip(self.pathtofile, self.pathtocat) :

            # path to files for galaxies
            wcs_in = self.paths.wcs_dir.joinpath(cat.stem.replace('_ldac','_wcs.fits'))
            mask_in = self.paths.mask_dir.joinpath(cat.stem.replace('_ldac','_mask.fits'))
            unc_in = self.paths.unc_dir.joinpath(cat.stem.replace('_ldac','_unc.fits'))
            
            in_names = [str(wcs_in), str(mask_in), str(unc_in)]
            
            wcs_out = reproj_dir.joinpath(str(wcs_in.name).replace('.fits','_reproj.fits'))
            mask_out = reproj_dir.joinpath(str(mask_in.name).replace('.fits','_reproj.fits'))
            unc_out = reproj_dir.joinpath(str(unc_in.name).replace('.fits','_reproj.fits'))

            out_names = [str(wcs_out), str(mask_out), str(unc_out)]

            for in_name, out_name in zip(in_names, out_names) :

                print ('\nRunning SWARP')
                print('\t for %s ...'%in_name)
                
                kwargs = {
                    'cmd': 'swarp',
                    'config': {
                        'IMAGEOUT_NAME': out_name,
                        'SUBTRACT_BACK' :'N',
                        'RESCALE_WEIGHTS':'N',
                        'GAIN_DEFAULT':'1.0',
                        'CENTER_TYPE' : 'MANUAL',
                        'CENTER':'%s,%s'%(image_centre[0],image_centre[1]),
                        'PIXEL_SCALE': '%s'%pixel_scale,
                        'PIXELSCALE_TYPE': 'MANUAL',
                        'IMAGE_SIZE':'%s,%s'%(image_size[0],image_size[1]),
                        'NTHREADS': '%s'%nthrds
                    }
                }
                        
                # Build the command to run SWARP
                self.cmd_swarp = self.build_cmd(in_name, **kwargs)
                
                # Run SWARP
                os.system (self.cmd_swarp)
            
                # merge and update the header after detrend (before projection) with the common projection header
                detrend_hdr = fits.open(str(file))[0].header
                
                reproj_data = fits.open(out_name)[0].data
                #reproj_hdr = fits.open(out_name)[0].header
                
                # update the detrend header with that of the reprojected image, but mantaining the exptime card'
                et_value = detrend_hdr['EXPTIME']
                et_comment = detrend_hdr.comments['EXPTIME']
                detrend_hdr.extend(reproj_hdr.cards, strip=False, update=True)
                detrend_hdr.set('EXPTIME', et_value, et_comment)  
                
                print ('\nMerging the headers before and after reprojection in\n%s\n'%out_name) 
                fits.writeto(out_name, data=reproj_data, header=detrend_hdr, overwrite=True)
            



class ProjImage (object) :
    

    def __init__ (self, reproj_filename, fwhm=5.0) :
        
        self.filename = reproj_filename
        name = str(reproj_filename.stem)
        pos = name.find('_')   
        self.obj = name[:pos]
        self.data = fits.open(str(reproj_filename))[0].data
        self.hdr = fits.open(str(reproj_filename))[0].header
        self.wcs = WCS(fits.open(str(reproj_filename))[0].header)
        self.fwhm = fwhm

        self.checkplots_path = Path('.').joinpath('checkplots')
        self.checkplots_path.mkdir(parents=True, exist_ok=True)


    def find_image_stars (self, find_stars=IRAFStarFinder, threshold_nstd=50., peakmax=None) :
        '''
        This method computes the image threshold to find stars with 
        IRAFFindStars algorithm. An astropy Table is created with pixel positions. 
        '''
        
        # find stars in the input image
        sigclip_sigma=2.0  
        sigclip_iters=5
        image_mean, image_median, image_std = sigma_clipped_stats(self.data, sigma=sigclip_sigma, maxiters=sigclip_iters)    
        print (image_mean, image_median, image_std)
        threshold = image_median + (threshold_nstd * image_std)
        print('Threshold for finding bright stars', threshold)
        
        iraffind = find_stars(threshold=threshold, fwhm=self.fwhm, peakmax=peakmax)
        self.tbl = iraffind(self.data)
        self.tbl.keep_columns(['xcentroid', 'ycentroid'])

        print('\nNumber of stars found in the image : %s\n' %len(self.tbl))


    def match_with_gaia (self, gaia_stars) :
        '''
        Centroid positions are converted in sky coordinates, by means of 
        the image wcs solution. Gaia DR2 stars are then retreived to be matched.
        '''
        # Convert pixel positions in coordinates
        ra, dec = self.wcs.all_pix2world(self.tbl['xcentroid'], self.tbl['ycentroid'], 0)
        self.coord = SkyCoord (ra=ra*u.deg, dec=dec*u.deg)
        
        # .. and update tbl
        ra = Column(self.coord.ra, name='ra')
        dec = Column(self.coord.dec, name='dec')
        self.tbl.add_column(ra)
        self.tbl.add_column(dec)
        
        # match with Gaia stars
        idxc, idxcatalog, d2d, d3d = gaia_stars.coord.search_around_sky(self.coord, 1.*u.arcsec)
        print('\nNumber of matching stars between the image and Gaia : %s\n' %len(idxc)) 
        
        # update tbl
        self.tbl = self.tbl[idxc]
        gaia_stars.coord = gaia_stars.coord[idxcatalog]
        gaia_ra = Column(gaia_stars.coord.ra, name='gaia_ra')
        gaia_dec = Column(gaia_stars.coord.dec, name='gaia_dec')
        gaia_r = Column(gaia_stars.rmag[idxcatalog], name='gaia_r_mag')
        gaia_r_err = Column(gaia_stars.e_rmag[idxcatalog], name='gaia_r_mag_err')
        gaia_BP_RPmag = Column(gaia_stars.BP_RPmag[idxcatalog], name='gaia_BP_RPmag')
        gaia_BP_RPmag_err = Column(gaia_stars.e_BP_RPmag[idxcatalog], name='gaia_BP_RPmag_err')
        self.tbl.add_column(gaia_ra)
        self.tbl.add_column(gaia_dec)
        self.tbl.add_column(gaia_r)
        self.tbl.add_column(gaia_r_err)
        self.tbl.add_column(gaia_BP_RPmag)
        self.tbl.add_column(gaia_BP_RPmag_err)
        
        for col in self.tbl.colnames:
            self.tbl[col].info.format = '%.8f'
 

    def img_psf_fitting (self, doplot) :
        
        # Create a table with the centroids pixel positions
        stars_tbl = Table()
        stars_tbl['x'] = self.tbl['xcentroid']
        stars_tbl['y'] = self.tbl['ycentroid']
        
        # define nddata object and extract the ePSF stars
        nddata = NDData(data=self.data)
        self.extracted_stars = extract_stars(nddata, stars_tbl, size=25)
        
        # 2 plots of 25 stars
        if doplot :
            
            nrows = 5
            ncols = 5

            for j in range(2) :
                
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 7), squeeze=True)
                
                ax = ax.ravel()
                
                zscale = ZScaleInterval()
            
                for i in range(nrows*ncols*j, nrows*ncols*(j+1)):

                    if i < 25 :
                        k = i
                    else:
                        k = i - 25
                
                    if i <= len(self.extracted_stars)-1 :
                        #vmin, vmax = zscale.get_limits(stars[i].data)
                        #ax[i].imshow(stars[i].data, origin='lower', cmap='viridis')#, vmin=vmin, vmax=vmax)
                        norm = simple_norm(self.extracted_stars[i], 'log', percent=99.)
                        ax[k].imshow(self.extracted_stars[i], origin='lower', cmap='viridis', norm=norm)
                        ax[k].set_title('star %s'%(i+1), fontsize=12)
                        ax[k].axes.get_xaxis().set_visible(False)
                        ax[k].axes.get_yaxis().set_visible(False)
                    else:
                        ax[k].axis('off')
                
                fig.suptitle(self.obj)
            
                plt.subplots_adjust (left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.3, hspace=0.3)
                
                plt.show()
                
                fig.savefig(self.checkplots_path.joinpath(str(self.filename.name).replace('.fits','_stars_%s.png'%(j+1))))   
                
                plt.close()
        
        # Build the ePSF
        epsf_builder = EPSFBuilder(oversampling=1, maxiters=10, progress_bar=True)
        epsf, fitted_stars = epsf_builder(self.extracted_stars)

        # Plot the resulting ePSF
        if doplot :
            fig, ax = plt.subplots (figsize=(6.92, 7))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            vmin, vmax = zscale.get_limits(epsf.data)
            im = ax.imshow(epsf.data, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, cax = cax)
            cbar.mappable.set_clim(vmin, vmax)
            ax.set_title(self.obj)
            plt.subplots_adjust (left=0.10, right=0.75, bottom=0.10, top=0.90)
            #plt.show()
            fig.savefig(self.checkplots_path.joinpath(str(self.filename.name).replace('.fits','_epsf.png')))
            plt.close()   
        #fig.savefig(self.filename.parent.joinpath('%s_psf.png'%self.filename.stem))
        #psf_model_filename = str(self.filename.parent.joinpath('%s_psf.fits'%self.filename.stem))
        #fits.writeto(psf_model_filename, epsf.data, overwrite=True)

        # Update tbl with fitted centroid positions and fluxes
        x_fit = Column(fitted_stars.center[:,0], name='x_fit')
        y_fit = Column(fitted_stars.center[:,1], name='y_fit')
        self.tbl.add_column(x_fit)
        self.tbl.add_column(y_fit)
        self.tbl['x_fit'].info.format = '%.8f'
        self.tbl['y_fit'].info.format = '%.8f'
        

        # Add the conversion to ra dec
        ra_fit, dec_fit = self.wcs.all_pix2world(self.tbl['x_fit'], self.tbl['y_fit'], 0)
        coord = SkyCoord (ra=ra_fit*u.deg, dec=dec_fit*u.deg)
        ra_fit = Column(coord.ra, name='ra_fit')
        dec_fit = Column(coord.dec, name='dec_fit')
        self.tbl.add_column(ra_fit)
        self.tbl.add_column(dec_fit)
        self.tbl['ra_fit'].info.format = '%.8f'
        self.tbl['dec_fit'].info.format = '%.8f'



class GAIAStars (object) :
    
    def __init__ (self, obj, my_catname = 'Gaia-DR2') : 
        
        self.obj = obj
        self.catname = my_catname
        
        
    def gaia_query (self, lmag=12, hmag=19) :
        
        #Vizier.ROW_LIMIT = -1
        #Vizier.TIMEOUT = 60 # sets the timeout to 60s

        new_limit = 60  # sets the timeout to 60s
        with conf.set_temp('remote_timeout', new_limit):
            obj_coord = SkyCoord.from_name(self.obj)
        
        vizier = Vizier(columns=['*', 'E(BR/RP)'], catalog='Gaia-DR2', row_limit=-1, timeout=60)
        result = vizier.query_region(obj_coord, radius=1*u.deg) 

        #result = Vizier.query_region(obj_coord, radius=1*u.deg, catalog=self.catname)
        # cuts on bright and faint star magnitudes
        mask1 = (result[0]['Gmag']>lmag)&(result[0]['Gmag']<hmag)
        # cuts on colours as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        mask2 = (result[0]['BPmag']-result[0]['RPmag']>-0.5)&(result[0]['BPmag']-result[0]['RPmag']<2.75)
        # check of consistency for the excess factor as in https://www.aanda.org/articles/aa/full_html/2018/08/aa32756-18/F17.html
        mask3 = (result[0]['E_BR_RP_']<1.3+0.06*(result[0]['BPmag']-result[0]['RPmag'])**2.)
        
        #gaia_tbl = result[0][mask1&mask2]
        gaia_tbl = result[0][mask1&mask2&mask3]

        self.Gmag = gaia_tbl['Gmag']
        self.e_Gmag = gaia_tbl['e_Gmag']
        self.BP_RPmag = gaia_tbl['BPmag'] - gaia_tbl['RPmag']
        self.e_BP_RPmag = np.sqrt(gaia_tbl['e_BPmag']**2.+gaia_tbl['e_RPmag']**2)
        '''
        Transforamtion as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.T8 
        G - r = - 0.12879 + 0.24662*(BP-RP) - 0.027464*(BP-RP)^2 - 0.049465*(BP-RP)^3 
        '''
        self.rmag = self.Gmag + 0.12879 - 0.24662*(self.BP_RPmag) + 0.027464*(self.BP_RPmag)**2. + 0.049465*(self.BP_RPmag)**3.
        self.e_rmag = np.sqrt(self.e_Gmag**2. + 0.066739**2.)
        
        self.coord = SkyCoord(ra=gaia_tbl['RA_ICRS'], dec=gaia_tbl['DE_ICRS'])
        

if __name__ == "__main__":
     
    start_time = timeit.default_timer()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('pathtodata', help='path to observation date folder')
    parser.add_argument('object', help='target name')

    parser.add_argument('--plot', action="store_true", default=False, help='make check plots')
    parser.add_argument('--overwrite_wcs', action="store_true", default=False, help='overwrite files in wcs_corrected, mask and uncertainty folders (test phase)')
    parser.add_argument('--rotate_image', action="store_true", default=False, help='perform 90 degree rotation in the rough wcs solution')
    parser.add_argument('--from_name', action="store_true", default=False, help='retrieve object coordinates from CDS for the rough wcs solution')
    parser.add_argument('--exclude', nargs='+', default=[], type=str, help='list of objects to exclude from the astrometry')
    
    args = parser.parse_args()
   
    # load list of images from the given path
    im_path = ImagesPaths (Path(args.pathtodata))
    im_path.make_dirs()
    #images = im_path.make_imlist()

    keys = ['OBJECT', 'FILTER']
    images = ImageFileCollection(im_path.path.joinpath('output','detrend'), keywords=keys)

    match = images.summary['OBJECT'] == args.object
    files = images.summary['file'][match]
    
    # Catch the error if an object was removed from the detrended files but not from the input list
    # i.e. the object name is in myobjects list but not in the ImageFileCollection
    if len(files) == 0 :
        sys.exit('\nSome file is missing for %s!\n'%args.object)

    astr_imgs = AstrometricImages (im_path, files)

    # Sort the list of files in the way that humans expect (human sorting)
    astr_imgs.files = astr_imgs.sort_nicely(list(astr_imgs.files))
    
    astr_imgs.init_paths()
    
    astr_imgs.rough_wcs(args.rotate_image, args.from_name)
    
    if args.overwrite_wcs :
        astr_imgs.clean_files()
    
    astr_imgs.run_sextractor()
    astr_imgs.filter_catalog(args.exclude)
    astr_imgs.run_scamp()
    
    astr_imgs.save_wcs()
    
    astr_imgs.run_swarp()
    
    astr_imgs.astr_precision(args.plot)
    
    astr_imgs.reproj_single_frame()
    

