# coding: utf-8

import os

import timeit

import numpy as np

from pathlib import Path

from astropy.io import fits

from matplotlib import pyplot as plt


class SatPixMasks(object) :
    

    def __init__ (self, list_of_frames) :
        
        self.list_of_frames = list_of_frames
    
    
    def set_new_masks (self) :

        self.sat_pix_mask_fits = []
        
        for i, frame in enumerate(self.list_of_frames) :
            
            hdulist = fits.open(frame)
            
            data = hdulist[0].data
            
            hdr = hdulist[0].header

            # determine the gain from ccdproc meta data
            gaincor = hdr['GAINCOR']
            gainpos = gaincor.find('gain=') + len('gain=')
            gainend = gaincor.find('electron') - 1
            gain = float(gaincor[gainpos:gainend])
            
            # Create the saturated pixel mask
            exptime = hdr['EXPTIME']
            factor = 40000 * gain / exptime
            sat_pix_mask = np.full(data.shape, 99.)
            sat_pix_mask[data>factor] = 1e12
           
            # Save saturated pixels mask
            sat_pix_mask_filename = str(frame).replace('frame','sp_mask_frame')
            
            print('\nSaving saturated pixels mask in\n%s\n'%sat_pix_mask_filename)
            fits.writeto(sat_pix_mask_filename, data=sat_pix_mask, header=hdr, overwrite=True)

            self.sat_pix_mask_fits.append(sat_pix_mask_filename)
        


    def remove_new_masks (self) :

        for fits in self.sat_pix_mask_fits :
            
            os.remove(fits)

            print('%s removed!'%fits)
            

class ReprojectedImage (object) :


    def __init__ (self, filename) :
        
        self.filename = filename
        self.hdr = fits.open(filename)[0].header


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

    
    def reproj_masks (self, mask_fits, nthrds = 0) :
        '''
        Method to perform the reprojecton of the new mask images onto the common place-folder for all the CCDs. 
        '''

        # Commoon projection parameters
        image_size = [self.hdr['NAXIS1'],self.hdr['NAXIS2']]
        image_centre = [self.hdr['CRVAL1'],self.hdr['CRVAL2']]
        pixel_scale = float(self.hdr['CD2_2'])*3600

        self.out_name = self.filename.replace('.fits','_sp_mask.fits')

        print ('\nRunning SWARP for ...')
        for fits in mask_fits :
            print(fits)
        
        kwargs = {
            'cmd': 'swarp',
            'config': {
                'IMAGEOUT_NAME': self.out_name,
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
        self.cmd_swarp = self.build_cmd(mask_fits, **kwargs)
                
        # Run SWARP
        os.system (self.cmd_swarp)

        
        
if __name__ == "__main__":
    
    start_time = timeit.default_timer()
    
    galnames, paths = np.loadtxt('paths_to_uncalib_WFI.txt', dtype='str', unpack=True)
    
    galnames = np.atleast_1d(galnames)
    paths = np.atleast_1d(paths)
    
    for gal, p in zip(galnames, paths) :

        p = Path(p)
         
        path_to_wcs = p.parent.joinpath('wcs_corrected')
         
        for filt in ['Rc', 'Ha'] :
        
            #if filt == 'Rc' :
            #    continue

            for i in range(43) :

                #if (i!=0) * (i!=1) :
                #    continue

                ccd_str = ''

                for j in range(8) :
                    if path_to_wcs.joinpath('wcs%s_%s_%s_frame%s.fits'%(gal,filt,j+1,i+1)).exists() :
                        ccd_str = ccd_str + str(j+1)
                
                if ccd_str == '' :
                    continue

                list_of_frames = sorted(path_to_wcs.glob('wcs%s_%s_[%s]_frame%s.fits'%(gal,filt,ccd_str,i+1)))

                spm = SatPixMasks(list_of_frames)
                
                spm.set_new_masks()

                repr_file = p.joinpath('%s_%s_stack%s.fits'%(gal,filt,i+1))
                
                if not repr_file.exists() :
                    continue
                 
                repr_img = ReprojectedImage(str(repr_file))
                 
                repr_img.reproj_masks(spm.sat_pix_mask_fits)
                 
                spm.remove_new_masks()


    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print ('Process terminated in %d:%02d:%02d'%(h, m, s))
