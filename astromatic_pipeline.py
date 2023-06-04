# coding: utf-8
'''
This code is modified with respect to the original pipeline.
The astromatic_wrapper package is removed for compatibility 
with python 3.

SCAMP version 2.10 is employed where Gaia-DR2 and Gaia-EDR3
are allowed as reference catalog.

The call to refine_astronemy routine is eliminated. 
'''

import os
import datetime
import ccdproc
import numpy as np
from astropy.io import fits

from pathlib import Path

from glob import glob
import sys
import ast

from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from matplotlib import pyplot as plt
import matplotlib.cm as cm


def init_aw_pipeline (obsdir) :

    # Set defaults
    base_path = os.path.join (obsdir, 'processed', 'output')
    
    paths = {
        'temp': os.path.join(base_path, 'temp'),
        'wcs_corrected': os.path.join(base_path, 'wcs_corrected'),
        'mask': os.path.join(base_path, 'mask'),
        'var': os.path.join(base_path, 'var'),
        'catalogs': os.path.join(base_path, 'catalogs'),
        'stacks': os.path.join(base_path, 'stacks'),
        'images': base_path
    }

    for key in paths :
        print('\nCreating path %s ...\n'%paths[key])
        try:
            os.makedirs(paths[key])
        except OSError:
            pass

    return paths


def init_exposures (paths, objectname, excl_ccdlist) :
    
    # Selection of ccd to include in the mosaic
    incl_ccdlist = [i+1 for i in range(8) if (i+1) not in excl_ccdlist]
    str_ccd = ''
    for ccd in incl_ccdlist :
        str_ccd = str_ccd + str(ccd)
    
    # create a list of input files for the astromatic pipeline
    
    exposures = [
        {
            'image': exp,
        }
        for exp in sorted(glob(os.path.join(paths['images'],'*%s_[%s]_*'%(objectname,str_ccd))))
    ]

    return exposures


def build_cmd (filenames, **kwargs):
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


def refine_astrometry (paths, ref_catalog) :
    
    # search for the last reference star catalog downloaded
    list_of_ref = glob(os.path.join(paths['catalogs'],'%s*'%ref_catalog)) 
    latest_ref = max(list_of_ref, key=os.path.getctime)
    print ('Last reference catalog downloaded : %s' %latest_ref)

    # script to convert the FITS_LDAC reference catalog (saved by SCAMP) to a data file as ldactoasc SExtractor command 
    ref_cat_file = os.path.join(paths['catalogs'], latest_ref)
    ref_dat_file = ref_cat_file.replace('.cat','.dat')
    hdu = fits.open(ref_cat_file)[2]
    header = hdu.header
    data = hdu.data
    print ('Saving %s ...'%ref_dat_file)
    with open(ref_dat_file, 'w') as f :
        f.write('#\t 1 %s\t %s\t [%s]\n' %(header['TTYPE1'],header.comments['TTYPE1'],header['TUNIT1']))
        f.write('#\t 2 %s\t %s\t [%s]\n' %(header['TTYPE2'],header.comments['TTYPE2'],header['TUNIT2']))
        f.write('#\t 3 %s\t %s\t [%s]\n' %(header['TTYPE3'],header.comments['TTYPE3'],header['TUNIT3']))
        f.write('#\t 4 %s\t %s\t [%s]\n' %(header['TTYPE4'],header.comments['TTYPE4'],header['TUNIT4']))
        f.write('#\t 5 %s\t %s\t [%s]\n' %(header['TTYPE5'],header.comments['TTYPE5'],header['TUNIT5']))
        f.write('#\t 6 %s\t %s\t [%s]\n' %(header['TTYPE6'],header.comments['TTYPE6'],header['TUNIT6']))
        f.write('#\t 7 %s\t %s\t [%s]\n' %(header['TTYPE7'],header.comments['TTYPE7'],header['TUNIT7']))
    with open(ref_dat_file, 'a') as f :
        for line in data :
            f.write('%.6f\t %.6f\t %.6e\t %.6e\t %.2f\t %.2f\t %.1f\n' %(line[0], line[1], line[2], line[3], line[4], line[5], line[6]))

    # load registered and reference radec coordinates (from registered image and the reference catalog) 
    img_dat_file = os.path.join(paths['catalogs'], 'scamp_1.cat')
    ra_img, dec_img = np.loadtxt(img_dat_file, usecols=(3,4), unpack=True)
    ra_cat, dec_cat, mag_cat = np.loadtxt(ref_dat_file, usecols=(0,1,4), unpack=True)

    # match between registered and reference coordinates by searching around a circle of 5 arcsecs
    img_coord = SkyCoord (ra=ra_img, dec=dec_img, unit=u.deg)
    cat_coord = SkyCoord (ra=ra_cat, dec=dec_cat, unit=u.deg)
    idximg, idxcat, d2d, d3d  = cat_coord.search_around_sky (img_coord, 5*u.arcsec)    # d2d is the correct angular separation
    print ('Number of matching stars \nbetween reference %s and registered by SCAMP : %s' %(ref_catalog,len(idximg)))

    # mean of angular distance residuals, standard deviation and error of the mean
    print ('*********************************************************')
    print ('Mean, standard deviation and error of angular distances :')
    print (np.mean(d2d.arcsec), np.std(d2d.arcsec), np.std(d2d.arcsec)/np.sqrt(len(d2d.arcsec)))
    flt = (mag_cat[idxcat]>0)*(mag_cat[idxcat]<20)
    print ('Mean, standard deviation and error of %s angular distances (magnitude cut off) :\n' %len(d2d[flt]))
    print (np.mean(d2d[flt].arcsec), np.std(d2d[flt].arcsec), np.std(d2d[flt].arcsec)/np.sqrt(len(d2d[flt].arcsec)))
    print ('*********************************************************')

    return img_coord, cat_coord, idximg, idxcat
    

def plot_astrometry (stack_filename, img_coord, cat_coord, idximg, idxcat) :

    # Transform coordinate to pixel with SWarp wcs solution    *** header with no distorsion coefficients ***
    hdu = fits.open(stack_filename)
    hdr = hdu[0].header
    im = hdu[0].data
    #print (repr(header))    # OK. It doesn' have the conflicting Pi_j header cards!
    # wcs initialization
    wcs_header = WCS(hdr)
    # wcs to pixel transformation
    x_img, y_img = skycoord_to_pixel(img_coord[idximg], wcs_header)
    x_cat, y_cat = skycoord_to_pixel(cat_coord[idxcat], wcs_header)

    # Conditions to discriminate stars belonging to different chips by coordinates/pixels
    image_size = [float(hdr['NAXIS1']),float(hdr['NAXIS2'])]
    chip1 = (x_cat<image_size[0]/4)*(y_cat>image_size[1]/2)
    chip2 = (x_cat>image_size[0]/4)*(x_cat<image_size[0]*2/4)*(y_cat>image_size[1]/2)
    chip3 = (x_cat>image_size[0]*2/4)*(x_cat<image_size[0]*3/4)*(y_cat>image_size[1]/2)
    chip4 = (x_cat>image_size[0]*3/4)*(y_cat>image_size[1]/2)
    chip5 = (x_cat<image_size[0]/4)*(y_cat<image_size[1]/2)
    chip6 = (x_cat>image_size[0]/4)*(x_cat<image_size[0]*2/4)*(y_cat<image_size[1]/2)
    chip7 = (x_cat>image_size[0]*2/4)*(x_cat<image_size[0]*3/4)*(y_cat<image_size[1]/2)
    chip8 = (x_cat>image_size[0]*3/4)*(y_cat<image_size[1]/2)

    # Astrometry: Check plot 1 

    fig, ax = plt.subplots (2, 2, sharex='col', sharey='row', figsize=(10,10))
    fig.subplots_adjust(wspace = 0.05, hspace = 0.05)
    # spherical offset from image registered positions to reference positions
    dra, ddec = img_coord[idximg].spherical_offsets_to(cat_coord[idxcat])
    x0 = img_coord[idximg].ra.arcsec
    y0 = img_coord[idximg].dec.arcsec
    
    ax[0,0].scatter(x0, dra.arcsec, marker = 'o', linewidth=0.5, facecolors = 'none', color = 'black')
    ax[0,0].set_xlim(np.min(x0)-100,np.max(x0)+100)
    ax[0,0].set_ylim(-1,+1)
    ax[0,0].set_ylabel(r'$\alpha_\mathrm{cat}$ $-$ $\alpha_\mathrm{img}\,\mathrm{[arcsec]}$', fontsize=16)
    ax[0,1].scatter(y0, dra.arcsec, marker = 'o', linewidth=0.5, facecolors = 'none', color = 'black')
    ax[0,1].set_xlim(np.min(y0)-100,np.max(y0)+100)
    ax[1,0].scatter(x0, ddec.arcsec, marker = 'o', linewidth=0.5, facecolors = 'none', color = 'black')
    ax[1,0].set_xlabel(r'$\alpha_\mathrm{img}\,\mathrm{[arcsec]}$', fontsize=16)
    ax[1,0].set_ylabel(r'$\delta_\mathrm{cat}$ $-$ $\delta_\mathrm{img}\,\mathrm{[arcsec]}$', fontsize=16)
    ax[1,0].set_ylim(-1,+1)
    ax[1,1].scatter(y0, ddec.arcsec, marker = 'o', linewidth=0.5, facecolors = 'none', color = 'black')
    ax[1,1].set_xlabel(r'$\delta_\mathrm{img}\,\mathrm{[arcsec]}$', fontsize=16)

    # Astrometry: Check plot 2

    fig, ax = plt.subplots (1, 1, figsize=(8,8))
    
    #x1 = cat_coord[idxcat].ra.arcsec
    #y1 = cat_coord[idxcat].dec.arcsec
    
    ax.quiver(x0, y0, dra.arcsec*100, ddec.arcsec*100, angles='xy', scale_units='xy', scale=1)    # angles='xy', scale_units='xy', scale=1 having the same units as x and y 
    #ax.quiver(x0, y0, x1-x0, y1-y0, units='xy')
    ax.set_xlim(np.min(x0)-300,np.max(x0)+300)
    ax.set_ylim(np.min(y0)-300,np.max(y0)+300)
    ax.set_xlabel(r'$\alpha_\mathrm{img}\,\mathrm{[arcsec]}$', fontsize=16)
    ax.set_ylabel(r'$\delta_\mathrm{img}\,\mathrm{[arcsec]}$', fontsize=16)

    # Astrometry: Check plot 3 

    fig, ax = plt.subplots (1, 1, figsize=(8,8))
    
    #ax.scatter(mag_cat[idxcat], d2d.arcsec, marker = 'o', linewidth=0.5, facecolors = 'none', color = 'black')
    
    colors = cm.rainbow(np.linspace(0, 1, 8))
    
    ax.scatter(mag_cat[idxcat][chip1], d2d.arcsec[chip1], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[0], label='chip1')
    ax.scatter(mag_cat[idxcat][chip2], d2d.arcsec[chip2], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[1], label='chip2')
    ax.scatter(mag_cat[idxcat][chip3], d2d.arcsec[chip3], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[2], label='chip3')
    ax.scatter(mag_cat[idxcat][chip4], d2d.arcsec[chip4], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[3], label='chip4')
    ax.scatter(mag_cat[idxcat][chip5], d2d.arcsec[chip5], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[4], label='chip5')
    ax.scatter(mag_cat[idxcat][chip6], d2d.arcsec[chip6], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[5], label='chip6')
    ax.scatter(mag_cat[idxcat][chip7], d2d.arcsec[chip7], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[6], label='chip7')
    ax.scatter(mag_cat[idxcat][chip8], d2d.arcsec[chip8], marker = 'o', linewidth=0.8, facecolors = 'none', color=colors[7], label='chip8')
    
    ax.set_xlim(15.5,20.5)
    ax.set_xlabel(r'$m_\mathrm{cat}$', fontsize=16)
    ax.set_ylabel('angular separation [arcsec]', fontsize=16)
    ax.legend(loc=2, frameon=False, scatterpoints=1)

    plt.show()
    



def astrometry (paths, config, exposures, stack_name, ref_catalog, ref_band='DEFAULT', dist_degree = '3', nthrds = 0) :


    # Input data frames for SExtractor
    print ('Running SExtractor\n')
    print ('Input data frames for SExtractor are in\n%s\n' %os.path.dirname(exposures[0]['image']))
    
    # Generate catalogs from sextractor
    
    catalog_names = []
    
    for files in exposures:
        
        # Name of input data frame
        print (os.path.basename(files['image']))
        
        # Create names for the output catalogs for each image
        catalog_names.append(os.path.join(paths['catalogs'],
            os.path.basename(files['image']).replace('.fits', '_ldac.fits')))

        kwargs = {
            # Command to run SExtractor
            'cmd': 'sex',
            # Paramteres in the configuration file
            'config': {
                'CATALOG_NAME': catalog_names[-1],
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
            
        # Run SExtractor on data frame
        print ('\nCreating catalog %s ...' %os.path.basename(files['image']).replace('.fits', '_ldac.fits'))

        # Build the command to run SExtractor
        if config.getboolean('SExtractor', 'Execute') :
            
            cmd_sex = build_cmd(files['image'], **kwargs)
            
            print('\nExecuting the following command with SExtractor :\n%s'%cmd_sex)
            
            # Run SExtractor
            os.system (cmd_sex)
    

    # Filter the catalogs to improve the astrometry
    
    for i, catalog in enumerate(catalog_names) :
        
        hdulist = fits.open(catalog)
        tbldata = hdulist['ldac_objects'].data
        #mask = (tbldata['FLUX_AUTO']<=7000)*(tbldata['FLUX_AUTO']>=20)*(tbldata['FLAGS']==0)*(tbldata['FLUX_RADIUS']<=2)
        mask = (tbldata['FLUX_AUTO']<=config.getint('SExtractor', 'SourceMaxFlux'))*(tbldata['FLUX_AUTO']>=config.getint('SExtractor', 'SourceMinFlux'))*(tbldata['FLAGS']==0)*(tbldata['FLUX_RADIUS']<=config.getint('SExtractor', 'SourceFluxRadius'))
        newtbldata = tbldata[mask]
        tblhdu = fits.BinTableHDU (data=newtbldata, name='LDAC_OBJECTS')
        newhdulist = fits.HDUList ([hdulist['primary'], hdulist['ldac_imhead'], tblhdu])
        #print (hdulist.info())
        #print (newhdulist.info())
        newcatalog = catalog.replace('_ldac.fits','_newldac.fits')
        newhdulist.writeto(newcatalog, overwrite=True)
        catalog_names[i] = newcatalog
    
    
    # Get astrometric solution from SCAMP
    # Use SCAMP to get astrometric solutions
    
    mergedname = os.path.join(paths['catalogs'], 'scamp.cat') 
    fullname = os.path.join(paths['catalogs'], 'full.cat')
    
    checkplot_list = ['fgroups','distort','astr_interror2d','astr_interror1d','astr_referror2d','astr_referror1d','astr_chi2','psphot_error']
    checkplot_pathlist = [os.path.join(paths['catalogs'],checkplot) for checkplot in checkplot_list]
    checkplots = ','.join(checkplot_pathlist)
    nthrds = config.getint('Scamp', 'NThreads')

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
            'REFOUT_CATPATH': paths['catalogs'],
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
    cmd_scamp = build_cmd(catalog_names, **kwargs)

    print('\nExecuting the following command with SCAMP :\n%s'%cmd_scamp)
        
    # Run SCAMP
    print ('\nRunning SCAMP')
    print ('Finding astrometric solution ...\n')
    os.system (cmd_scamp)
    
    #img_coord, cat_coord, idximg, idxcat = refine_astrometry (paths, ref_catalog)   
 
    
    # Create wcs-corrected image frames, with updated headers, mask and error frames

    for files in exposures :
        
        # load image
        hdu = fits.open(files['image'])
        rwcs_data = hdu[0].data
        rwcs_header = hdu[0].header
        
        # import scamp output .head file
        newcat_file = os.path.join(paths['catalogs'], 
                                   os.path.basename(files['image']).replace('.fits', '_newldac.head'))
        
        # Solve the conflict with ascii encoding of the header text file by replacing 'é'' with 'e'
        with open(newcat_file, 'r') as file :
            filedata = file.read()
        newdata = filedata.replace('é', 'e')
        with open(newcat_file, 'w') as file :
            file.write(newdata)

        newcat_header = fits.Header.fromtextfile(newcat_file)
    
        # merge image header with scamp output header (updating the old header cards)
        rwcs_header.extend (newcat_header.cards, update=True)
        
        # save wcs files updating the image header
        wcsfile = os.path.join(paths['wcs_corrected'], os.path.basename(files['image']).replace('rwcs','wcs'))
        print ('Writing to ...\n %s' %wcsfile)
        fits.writeto (wcsfile, rwcs_data, header=rwcs_header, overwrite=True)
        
        # save mask frames
        maskfits = hdu['MASK']
        mask_temp = np.array(np.zeros(maskfits.data.shape), float)
        mask_temp[maskfits.data==1] = 1e12
        mask_temp[maskfits.data==0] = 99
        maskfits.data = mask_temp
        maskfile = os.path.join(paths['mask'], os.path.basename(wcsfile.replace('.fits','_mask.fits')))
        print ('Writing to ...\n %s' %maskfile)
        fits.writeto (maskfile, maskfits.data, header=rwcs_header, overwrite=True)
        
        ## save variance map CHECK frames
        #varfits = hdu['UNCERT']
        #varfits.data[maskfits.data==1] = 1e6
        #var_temp = varfits.data**2.
        #varfile = os.path.join(paths['var'], os.path.basename(wcsfile.replace('.fits','_var_check.fits')))
        #print ('Writing to ...\n %s' %varfile)
        #fits.writeto (varfile, var_temp.data, header=rwcs_header, overwrite=True)
        
        # save variance map frames
        stdfits = hdu['UNCERT']
        varfile = os.path.join(paths['var'], os.path.basename(wcsfile.replace('.fits','_var.fits')))
        print ('Writing to ...\n %s' %varfile)
        fits.writeto (varfile, data=stdfits.data**2., header=rwcs_header, overwrite=True)


    # Resample (rotate and scale) and combine (stack) images
    
    stack_filename = os.path.join(paths['stacks'], stack_name)
    kwargs = {
        'cmd': 'swarp',
        #'cmd': 'different/path/to/swarp',
        'config': {
            #'WEIGHT_TYPE': 'MAP_WEIGHT',
            #'WEIGHT_SUFFIX': '.wtmap.fits',
            'IMAGEOUT_NAME': stack_filename,
            #'WEIGHTOUT_NAME': stack_filename.replace('.fits','.wtmap.fits'),
            'SUBTRACT_BACK' :'N',
            'NTHREADS': '%s'%nthrds
        }
    }
    
        
    filenames = [os.path.join(paths['wcs_corrected'], os.path.basename(exp['image']).replace('rwcs','wcs')) for exp in exposures]
    
    # Build the command to run SWARP
    cmd_swarp = build_cmd(filenames, **kwargs)

    print('\nExecuting the following command with SWarp :\n%s'%cmd_swarp)

    # Run SWarp on the images to stack
    print ('Running SWarp')
    print ('Co-adding the images ...')
    os.system (cmd_swarp)

    #plot_astrometry(stack_filename, img_coord, cat_coord, idximg, idxcat)


def single_frame_mosaic (paths, config, exposures, stack_filename, obj, nframes, frames=[0]) :

    hdulist = fits.open(stack_filename)
    hdr = hdulist[0].header
    print (repr(hdr))
    image_size = [hdr['NAXIS1'],hdr['NAXIS2']]
    pixel_scale = float(hdr['CD2_2'])*3600
    image_centre = [hdr['CRVAL1'],hdr['CRVAL2']]
    print (pixel_scale)
    
    nthrds = config.getint('Scamp', 'NThreads')

    # Separate the images belonging to the same frames
    lists_of_exposures = []
    for i in range(nframes) :
        exps = []
        for j in range(i, len(exposures), nframes) :
            exps.append(exposures[j])
        lists_of_exposures.append(exps)

    # Resample (rotate and scale) and combine (stack) images

    for i,exps in enumerate(lists_of_exposures) :
        
        stack_name_exp = '%s_stack%s.fits'%(obj,i+1)
        stack_filename_exp = os.path.join(paths['stacks'], stack_name_exp)
        
        kwargs = {
            'cmd': 'swarp',
            #'cmd': 'different/path/to/swarp',
            'config': {
                #'WEIGHT_TYPE': 'MAP_RMS',
                #'WEIGHT_IMAGE': lists_of_weights[i],
                #'COMBINE_TYPE':'WEIGHTED',
                'IMAGEOUT_NAME': stack_filename_exp,
                #'WEIGHTOUT_NAME': stack_filename_exp.replace('.fits','_wtmap.fits'),
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
        
            
        #filenames = [exp['image'] for exp in exposures]
        filenames = [os.path.join(paths['wcs_corrected'], os.path.basename(exp['image']).replace('rwcs','wcs')) for exp in exps]
        
        # Build the command to run SWARP
        cmd_swarp = build_cmd(filenames, **kwargs)
        
        # Run SWarp on the images to stack
        print ('Running SWarp')
        print ('Co-adding the images ...')
        print ('%s ...' %stack_filename_exp)
        os.system (cmd_swarp)


    mask_names = []
    for files in exposures:
        # Name of input data frame
        #print (os.path.basename(files['image']))
        # Create names for the output catalogs for each image
        wcs_name = os.path.join(paths['mask'],
                                os.path.basename(files['image']).replace('rwcs','wcs'))
        mask_names.append(os.path.join(paths['mask'],
                                os.path.basename(wcs_name).replace('.fits','_mask.fits')))

    # Separate the mask maps belonging to the same frames
    lists_of_mask_maps = []
    for i in range(nframes) :
        maskmaps = []
        for j in range(i, len(mask_names), nframes) :
            maskmaps.append(mask_names[j])    
        lists_of_mask_maps.append(maskmaps)

    # Resample (rotate and scale) and combine (stack) variance maps

    for i,exps in enumerate(lists_of_mask_maps) :
    
        stack_name_mask = '%s_stack%s_mask.fits'%(obj,i+1)
        stack_filename_mask = os.path.join(paths['stacks'], stack_name_mask)
        
        kwargs = {
            'cmd': 'swarp',
            #'cmd': 'different/path/to/swarp',
            'config': {
                #'WEIGHT_TYPE': 'MAP_RMS',
                #'WEIGHT_IMAGE': lists_of_weights[i],
                #'COMBINE_TYPE':'WEIGHTED',
                'IMAGEOUT_NAME': stack_filename_mask,
                #'WEIGHTOUT_NAME': stack_filename_mask.replace('.fits','_wtmap.fits'),
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
        
            
        #filenames = [exp['image'] for exp in exposures]
        filenames = [os.path.join(paths['mask'], os.path.basename(mask)) for mask in exps]

        # Build the command to run SWARP
        cmd_swarp = build_cmd(filenames, **kwargs)
        
        # Run SWarp on the images to stack
        print ('Running SWarp')
        print ('Co-adding the images ...')
        print ('%s ...' %stack_filename_mask)
        os.system (cmd_swarp)


    var_names = []
    for files in exposures:
        # Name of input data frame
        #print (os.path.basename(files['image']))
        # Create names for the output catalogs for each image
        wcs_name = os.path.join(paths['var'],
                                os.path.basename(files['image']).replace('rwcs','wcs'))
        var_names.append(os.path.join(paths['var'],
                                os.path.basename(wcs_name).replace('.fits','_var.fits')))


    # Separate the variance maps belonging to the same frames
    lists_of_var_maps = []
    for i in range(nframes) :
        varmaps = []
        for j in range(i, len(var_names), nframes) :
            varmaps.append(var_names[j])    
        lists_of_var_maps.append(varmaps)

    # Resample (rotate and scale) and combine (stack) variance maps

    for i,exps in enumerate(lists_of_var_maps) :
        
        stack_name_var = '%s_stack%s_var.fits'%(obj,i+1)
        stack_filename_var = os.path.join(paths['stacks'], stack_name_var)
        
        kwargs = {
            'cmd': 'swarp',
            #'cmd': 'different/path/to/swarp',
            'config': {
                #'WEIGHT_TYPE': 'MAP_RMS',
                #'WEIGHT_IMAGE': lists_of_weights[i],
                #'COMBINE_TYPE':'WEIGHTED',
                'IMAGEOUT_NAME': stack_filename_var,
                #'WEIGHTOUT_NAME': stack_filename_var.replace('.fits','_wtmap.fits'),
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
        
            
        #filenames = [exp['image'] for exp in exposures]
        filenames = [os.path.join(paths['var'], os.path.basename(var)) for var in exps]

        # Build the command to run SWARP
        cmd_swarp = build_cmd(filenames, **kwargs)
        
        # Run SWarp on the images to stack
        print ('Running SWarp')
        print ('Co-adding the images ...')
        print ('%s ...' %stack_filename_var)
        os.system (cmd_swarp)



def aw_pipeline (args, config) :
    
    # initialize the pipeline paths
    paths = init_aw_pipeline (os.path.join(args.pathtodata, args.obsdate))

    # list of fits file created with ImageFileCollection
    keys = ['object', 'filtname']
    images = ccdproc.ImageFileCollection(paths['images'], keywords=keys)

    # create the list of objects to reduce
    objs = [obj for obj in images.values('object', unique=True)]
    if args.object != [] :
        if set(args.object).issubset(objs) :
            objs = args.object
        else :
            print ('\n')
            sys.exit('Not a valid list of objects!\nInput a valid object for the astrometry and image stacking\n')
    
    # Excluding not working CCDs
    list_excl_ccd = ast.literal_eval(config.get('Image Stacking', 'excl_ccds'))

    num_ccd = 8

    filtnames = ['BB#Rc/162_ESO844', 'NB#Halpha/7_ESO856']

    for obj in objs :
        
        # update objs with obj_filt object definition
        for filtname in filtnames :
            
            match = (images.summary['object'] == obj) * (images.summary['filtname'] == filtname)
            #print (obj, filtname, len(images.summary['file'][match])/8, obj+'_%s'%filtname[3:5])
            obj_filt = obj+'_%s'%filtname[3:5]
            nframes = int(len(images.summary['file'][match])/num_ccd)
            
            if nframes > 0 :
                
                exposures = init_exposures (paths, obj_filt, list_excl_ccd)
                
                if args.astrometry :
                    print ('\n\nFinding astrometric solution for %s\n\n'%obj_filt)
                    astrometry (paths, config, exposures, '%s_stack.fits'%obj_filt, args.ref_catalog)
                
                if args.coadd :
                    print ('\n\nBuilding single frame moaics and stacking %s\n\n'%obj_filt)
                    single_frame_mosaic (paths, config, exposures, os.path.join(paths['stacks'], '%s_stack.fits'%obj_filt), obj_filt, nframes)




if __name__ == "__main__":

    class myswitches (object) :

        def __init__ (self, pathtodata = '/Users/arazza/DAS/Observations/MPG2.2m/DATA', obsdate = '2016-10-07', astrometry=True, coadd=True, obj=[]) :
            self.pathtodata = pathtodata
            self.obsdate = obsdate
            self.master = master
            self.badpixel = badpixel
            self.detrend = detrend 
            self.object = obj

    myargs = myswitches(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])

    aw_pipeline (myargs)

