# -*- coding: utf-8 -*-
"""
Created on Mon Apr 8 14:30 2019

@author: arazza
"""

import sys
import timeit
import argparse
import numpy as np
from shutil import move
from pathlib import Path
from configparser import ConfigParser
import ccdproc
from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt

class ImagesPaths (object) :
    
    # class for the creating and accessing the paths to reduction products
    def __init__ (self, path) :
        
        self.path = path
        self.indir = path.joinpath('input')
        self.interdir = path.joinpath('intermediate')
        self.masterdir = path.joinpath('masterframes')
        self.outdir = path.joinpath('output')
    
    # create the directories for the reduction products    
    def make_dirs (self) :
        
        self.indir.mkdir(parents=True, exist_ok=True)
        self.interdir.mkdir(parents=True, exist_ok=True)
        self.masterdir.mkdir(parents=True, exist_ok=True)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def move_to_indir (self) :
        
        list_of_fits = sorted(self.path.glob('*.fits'))
        
        for fits in list_of_fits :
            
            move(fits, self.indir.joinpath(fits.name))
    
    # create the list of files in the current directory with ImageFileCollection
    def make_imlist (self) :

        keys = ['OBJECT', 'FILTER']
        images = ccdproc.ImageFileCollection(self.indir, keywords=keys)
        
        return images


class CCDSections (object) :
    
    def __init__ (self, poly_model, overscan_section, trim_section) :
        
        self.poly_model = poly_model
        self.overscan_section = overscan_section
        self.trim_section = trim_section  


class SingleCCD (object) :
    '''
    Class defining the individual CCD to be reduced.
    '''
    def __init__ (self, file, paths, sections, ccd=None, gain=None, readout=None) :
        
        self.file = file
        self.paths = paths
        self.sections = sections
        self.ccd = ccdproc.CCDData.read(str(self.paths.indir.joinpath(file)),unit=u.adu)
        self.obj = self.ccd.header['OBJECT']
        self.filtname = self.ccd.header['FILTER']
        self.gain = self.ccd.header['EGAIN'] * u.electron / u.adu
        self.readout = self.ccd.header['ENOISE'] * u.electron

        
    def make_masterbias (self, images, domaster) :
        
        masterbias_file = self.paths.masterdir.joinpath('masterbias.fits')

        if (masterbias_file.exists()) * (not domaster) :
            
            print('\nLoading masterbias ...\n')
            self.masterbias = ccdproc.CCDData.read(str(masterbias_file), unit=u.adu)
        
        else :
            
            # creates the lists of bias frames
            match = (images.summary['OBJECT'] == 'Bias')
            biases = images.summary['file'][match]
            biaslist = []
            for bias in biases :
                print ('\nReducing bias %s ...\n' %bias)
                rawbias = self.paths.indir.joinpath(bias)
                ccd = ccdproc.CCDData.read(rawbias, unit=u.adu)
                ccd = ccdproc.subtract_overscan(ccd, overscan_axis=1, fits_section=self.sections.overscan_section, model=self.sections.poly_model)
                ccd = ccdproc.trim_image(ccd, fits_section=self.sections.trim_section)
                biaslist.append(ccd)
    
            print ('\nCreating masterbias ...\n')
            combiner = ccdproc.Combiner(biaslist)
            self.masterbias = combiner.median_combine()
            self.masterbias.write(str(masterbias_file), overwrite=True)
        
    
    def make_masterflat (self, images, domaster) :
        
        masterbias_file = self.paths.masterdir.joinpath('masterbias.fits')
        masterflat_file = self.paths.masterdir.joinpath('masterflat_%s.fits'%self.filtname)
        
        if (masterflat_file.exists()) * (not domaster) :
            
            print('\nLoading masterflat ...\n')
            self.masterflat = ccdproc.CCDData.read(str(masterflat_file), unit=u.adu)
        
        else :

            print('\nLoading masterbias ...\n')
            self.masterbias = ccdproc.CCDData.read(str(masterbias_file), unit=u.adu)
            
            # creates the lists of flatfield frames
            match = (images.summary['OBJECT'] == 'Skyflat') & (images.summary['FILTER'] == self.filtname)
            flats = images.summary['file'][match]
            
            flatlist = []
            for flat in flats :
                print ('\nReducing flat %s ...\n' %flat)
                rawflat = self.paths.indir.joinpath(flat)
                ccd = ccdproc.CCDData.read(rawflat, unit=u.adu)
                ccd = ccdproc.subtract_overscan(ccd, overscan_axis=1, fits_section=self.sections.overscan_section, model=self.sections.poly_model)
                ccd = ccdproc.trim_image(ccd, fits_section=self.sections.trim_section)
                ccd = ccdproc.subtract_bias(ccd, self.masterbias)
                ccd = ccdproc.create_deviation(ccd, gain=self.gain, readnoise=self.readout)
                ccd = ccdproc.gain_correct(ccd, self.gain)
                flatlist.append(ccd)
                
            print ('\nCreating masterflat ...\n')
            #scaling_func = lambda arr: 1/np.ma.average(arr)
            flatlist = [flat.divide(np.ma.average(flat.data)) for flat in flatlist]
            combiner = ccdproc.Combiner(flatlist)
            #combiner.scaling = scaling_func
            self.masterflat = combiner.median_combine()
            self.masterflat.write(str(masterflat_file), overwrite=True)
        
    
    def bad_pixel_mask (self, dobpm) :
        
        # Chcking if bad pixel mask npy file exists
        
        bpm_file = self.paths.masterdir.joinpath('bpm_%s.npy'%self.filtname)
        
        if (bpm_file.exists()) * (not dobpm) :
            
            self.bpm = np.load(bpm_file)
        
        else :

            # Checking if masterflat was already created
            masterflat_file = self.paths.masterdir.joinpath('masterflat_%s.fits'%self.filtname)
            if masterflat_file.exists() :
                print('\nLoading masterflat for bad pixel mask creation ...\n')
                self.masterflat = ccdproc.CCDData.read(str(masterflat_file), unit=u.adu)
                print ('\nCreating bad pixel mask for filter %s (it might take some time!) ...\n'%self.filtname)
                self.bpm = ccdproc.ccdmask (self.masterflat, findbadcolumns=True, ncmed=12, nlmed=12, ncsig=25, nlsig=25, lsigma=12, hsigma=12)
                # update bpm with a mask from low/high count dome flats
                self.bpm += np.load('bad_pixel_mask.npy')
                # update bpm with columns directly read from a txt file
                X, Y = np.loadtxt('bad_pixel_columns.txt', unpack=True)
                for i in range(len(X)) : 
                    self.bpm[int(Y[i]):,int(X[i])-1] = True 
                # Save bpm
                np.save(bpm_file, self.bpm)
            else :
                print('\nNo masterflat for filter %s to compute the bad pixel mask!\n'%self.filtname)


    def detrend (self, images, args) :
        
        # Loading (or creating) masterbias, masterflat and bad pixel mask
        masterbias_file = self.paths.masterdir.joinpath('masterbias.fits')
        if masterbias_file.exists() :
            print('\nLoading masterbias ...\n')
            self.masterbias = ccdproc.CCDData.read(str(masterbias_file), unit=u.adu)
        else :
            self.make_masterbias(images, True)

        masterflat_file = self.paths.masterdir.joinpath('masterflat_%s.fits'%self.filtname)
        if masterflat_file.exists() :
            print('\nLoading masterflat ...\n')
            self.masterflat = ccdproc.CCDData.read(str(masterflat_file), unit=u.adu)
        else :
            self.make_masterflat(images, True)
        
        bpm_file = self.paths.masterdir.joinpath('bpm_%s.npy'%self.filtname)
        if bpm_file.exists() :
            print('\nLoading bad pixel mask ...\n')
            self.bpm = np.load(bpm_file)
        else :
            self.bad_pixel_mask(True)

        ### DETREND ###
        '''
        1) Overscan subtraction and trimming
        2) Initialisation of the mask with the bad pixel mask, then updated with the saturated pixels and cosmic rays 
        3) Bias subtraction
        4) Error frame creation and Gain correction of the image (new units electrons instead of ADU)
        5) Flat field correction
        6) Normalization to electron per second
        '''
        print ('\nReducing frame %s ...'%self.file) 
        
        print ('\tOverscan subtraction ...')
        self.ccd = ccdproc.subtract_overscan(self.ccd, overscan_axis=1, fits_section=self.sections.overscan_section, model=self.sections.poly_model)
        if args.intermediate :
            self.ccd.write(self.paths.interdir.joinpath('o%s_%s.fits'%(self.obj,self.filtname)), overwrite=True)

        print ('\tTrimming the image ...')
        self.ccd = ccdproc.trim_image(self.ccd, fits_section=self.sections.trim_section)
        if args.intermediate :
            self.ccd.write(self.paths.interdir.joinpath('to%s_%s.fits'%(self.obj,self.filtname)), overwrite=True)
        
        # Update the mask with the bad pixel mask and saturated pixels
        self.ccd.mask = self.bpm
        self.ccd.mask[self.ccd.data>40000] = True   
        
        print ('\tBias subtraction ...')  
        self.ccd = ccdproc.subtract_bias(self.ccd, self.masterbias)
        if args.intermediate :
            self.ccd.write(self.paths.interdir.joinpath('bto%s_%s.fits'%(self.obj,self.filtname)), overwrite=True)
        
        print ('\tCreating error frame ...')
        self.ccd = ccdproc.create_deviation(self.ccd, gain=self.gain, readnoise=self.readout)
        
        print ('\tGain correction ...')
        self.ccd = ccdproc.gain_correct(self.ccd, self.gain)
        if args.intermediate :
            self.ccd.write(self.paths.interdir.joinpath('gbto%s_%s.fits'%(self.obj,self.filtname)), overwrite=True)
         
        print ('\tRemoving cosmic rays ...')    # Lacosmic technique, van Dokkum (2001) as implemented by McCully (2014).
        self.ccd = ccdproc.cosmicray_lacosmic(self.ccd, gain=1.0, sigclip=6.5, sigfrac=0.2, objlim=20.0, niter=4, gain_apply=False)
        if args.intermediate :
            self.ccd.write(self.paths.interdir.joinpath('crgbto%s_%s.fits'%(self.obj,self.filtname)), overwrite=True) 
        
        print ('\tFlat field correction ...')
        self.ccd = ccdproc.flat_correct(self.ccd, self.masterflat)
        #ccd_fc = ccd.divide(masterflat)
        #ccd_fc = ccdproc.CCDData(ccd_fc.data, uncertainty=ccd_fc.uncertainty, mask=ccd.mask, wcs=ccd.wcs, meta=ccd.meta, unit=u.electron)
        if args.intermediate :
            self.ccd.write(self.paths.interdir.joinpath('fcrgbto%s_%s.fits'%(self.obj,self.filtname)), overwrite=True)
        
        print ('\tNormalization to electron/second ...')
        #ccd_norm = self.ccd.divide(self.ccd.header['EXPTIME']*u.second)
        #self.ccd.data = ccd_norm.data
        #self.ccd.unit = ccd_norm.unit
        #self.ccd.uncertainty = ccd_norm.uncertainty
        self.ccd = self.ccd.divide(self.ccd.header['EXPTIME']*u.second, handle_mask='ff', handle_meta='ff')
        if args.intermediate :
            self.ccd.write(self.paths.interdir.joinpath('nfcrgbto%s_%s.fits'%(self.obj,self.filtname)), overwrite=True)
    
        # Plot of the results
        if args.plot :
            plt.figure()
            zscale = ZScaleInterval() 
            vmin, vmax = zscale.get_limits(self.ccd.data)
            plt.imshow(self.ccd.data, origin='lower', vmin=vmin, vmax=vmax, cmap='gray_r')
            plt.show()


    def save_detrended (self, num) :

        self.paths.detrend_dir = self.paths.outdir.joinpath('detrend')
        self.paths.detrend_dir.mkdir(parents=True, exist_ok=True)

        self.ccd.write(self.paths.detrend_dir.joinpath('%s_%s_%i.fits'%(self.obj,self.filtname,num+1)), overwrite=True)

        #detrend_list = sorted(self.paths.detrend_dir.glob('%s_%s_*'%(self.obj,self.filtname)))
        #
        #if detrend_list==[] :
        #    self.ccd.write(self.paths.detrend_dir.joinpath('%s_%s_1.fits'%(self.obj,self.filtname)), overwrite=True)
        #else :
        #    last_fname = detrend_list[-1].stem
        #    frame_index = int(last_fname[len(self.obj)+1+len(self.filtname)+1:-5])
        #    self.ccd.write(self.paths.detrend_dir.joinpath('%s_%s_%i.fits'%(self.obj,self.filtname,frame_index+1)), overwrite=True)


if __name__ == "__main__":

    start_time = timeit.default_timer()
    
    parser = argparse.ArgumentParser()

    parser.add_argument('pathtodata', help='path to observation date folder')
    parser.add_argument('object', help='target name')
    parser.add_argument('filter', help='filter name')

    parser.add_argument('-m', '--master', action="store_true", default=False, help='create master frames for detrending')
    parser.add_argument('-b', '--badpixel', action="store_true", default=False, help='force creating a bad pixel mask')
    parser.add_argument('-d', '--detrend', action="store_true", default=False, help='detrend the image, write a rough astrometric solution \
                            in the header, create a mask and an error frame for each CCD')
    parser.add_argument('-i', '--intermediate', action="store_true", default=False, help='save the intermediate products of image detrend')
    parser.add_argument('--plot', action="store_true", default=False, help='make check plots')
    
    args = parser.parse_args()

    # load list of images from the given path
    im_path = ImagesPaths (Path(args.pathtodata))
    im_path.make_dirs()
    im_path.move_to_indir()
    images = im_path.make_imlist()
    
    sections = CCDSections (poly_model=models.Legendre1D(14), overscan_section='[2049:2176,1:2176]', trim_section='[1:2048,1:2048]')

    # loop over the object files
    match = (images.summary['OBJECT'] == args.object) & (images.summary['FILTER'] == args.filter)
    files = images.summary['file'][match]
    
    for num, file in enumerate(files) :
        
        dpCCD = SingleCCD(file, im_path, sections) 
        
        if file == files[0] :
            dpCCD.make_masterbias(images, args.master)
            dpCCD.make_masterflat(images, args.master)
            dpCCD.bad_pixel_mask(args.badpixel)

        dpCCD.detrend(images, args)
        dpCCD.save_detrended(num)

    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('Process terminated in %d:%02d:%02d'%(hours, minutes, seconds))



    