# coding: utf-8

'''
Log of v2 changes:
- Bias frames are no longer gain corrected
- Errors in the Bias are not taken into account anymore (this is valid as long as we have a large number of biases)
- Re-corretcting the biases for the new empirical gain is no long necessary, but the step is kept to trim the bias frames.
- Bias subtraction for master and science frames is done before gain correction
'''

import argparse
from configparser import ConfigParser

import ccdproc
from astropy.io import fits
from astropy import units as u
from astropy.modeling import models
from astropy.wcs import WCS
import numpy as np
#import bottleneck as bn
import os
from glob import glob
import sys
import math as m
from shutil import copy
import matplotlib.pyplot as plt
import json
import re

from rough_wcs import load_object, first_wcs

#from astromatic_pipeline import aw_pipeline

from WFIconfig import ConfigWrite

def make_list (dir='.') :

    # create the list of files in the current directory with ImageFileCollection
    keys = ['object', 'eso obs targ name', 'eso dpr catg', 'eso dpr type', 'eso ins filt1 name']
    images = ccdproc.ImageFileCollection(dir, keywords=keys)
    
    # fill the masked values in column 'eso obs targ name'
    images.summary['eso obs targ name'].fill_value = 'NONE'
    images.summary['eso obs targ name'] = images.summary['eso obs targ name'].filled()

    return images


def janesick (config, rawdir, images, nhdu) :

    # creates the lists of bias and domeflat frames, then selecting a pair for each of them
    biases = images.summary['file'][images.summary['object'] == 'BIAS']
    if np.sum(images.summary['object'] == 'FLAT,SCREEN') > 0 :
        flats = images.summary['file'][images.summary['object'] == 'FLAT,SCREEN']
    else :
        flats = images.summary['file'][images.summary['object'] == 'FLAT,SKY']

    bias1 = ccdproc.CCDData.read(os.path.join(rawdir, biases[0]), hdu=nhdu, unit=u.adu).data
    bias2 = ccdproc.CCDData.read(os.path.join(rawdir, biases[1]), hdu=nhdu, unit=u.adu).data
    flat1 = ccdproc.CCDData.read(os.path.join(rawdir, flats[0]), hdu=nhdu, unit=u.adu).data   
    flat2 = ccdproc.CCDData.read(os.path.join(rawdir, flats[1]), hdu=nhdu, unit=u.adu).data

    # search for a 100x100 without bad/hot pixels, cosmic rays and strong gradients
    size = config.getint('Janesick', 'box_size')    # default 100
    bias_arr_of_max = ccdproc.block_reduce(bias1, size, func=np.max)
    bias_arr_of_min = ccdproc.block_reduce(bias1, size, func=np.min)
    #bias_min_diff = np.min(bias_arr_of_max-bias_arr_of_min)
    bias_min_diff_10 = np.percentile(bias_arr_of_max-bias_arr_of_min, 10)
    #print (bias_min_diff, bias_min_diff_10)
    flat_arr_of_max = ccdproc.block_reduce(flat1, size, func=np.max)
    flat_arr_of_min = ccdproc.block_reduce(flat1, size, func=np.min)
    #flat_min_diff = np.min(flat_arr_of_max-flat_arr_of_min)
    flat_min_diff_10 = np.percentile(flat_arr_of_max-flat_arr_of_min, 20)
    #print (flat_min_diff, flat_min_diff_10)
    bias_arr_diff = ccdproc.block_replicate(bias_arr_of_max-bias_arr_of_min, size, conserve_sum=False)
    flat_arr_diff = ccdproc.block_replicate(flat_arr_of_max-flat_arr_of_min, size, conserve_sum=False)
    # save boxes position equals to the minimum plus noise level
    boxes = []
    i=0

    while i <= bias_arr_diff.shape[0]-size :
        flag = False
        j=0
        while j <= bias_arr_diff.shape[1]-size :
            # print (i, j, bias_arr_diff[i,j], flag)
            if (bias_arr_diff[i,j]<=bias_min_diff_10)*(flat_arr_diff[i,j]<=flat_min_diff_10) :
                boxes.append([i,i+size,j,j+size])
                flag = True
                j+=size
            else :
                j+=1
        if flag :
            i+=size
        else :
            i+=1
    # medians of the values for each box
    gains = []
    readnoises = []
    for box in boxes :
        #print (np.median(arr_diff[box[0]:box[1],box[2]:box[3]]))
        b1 = bias1[box[0]:box[1],box[2]:box[3]].astype(np.int32)
        b2 = bias2[box[0]:box[1],box[2]:box[3]].astype(np.int32)
        f1 = flat1[box[0]:box[1],box[2]:box[3]].astype(np.int32)
        f2 = flat2[box[0]:box[1],box[2]:box[3]].astype(np.int32)
        
        dbias = b1 - b2
        dflat = f1 - f2

        gain = ((np.mean(f1)+np.mean(f2)) - (np.mean(b1)+np.mean(b2))) / (np.std(dflat)**2.-np.std(dbias)**2.)
        gains.append(gain)
        
        readnoise = gain * np.std(dbias) / m.sqrt(2)
        readnoises.append(readnoise)
        
        
        
    med_gain = np.median(np.array(gains))
    med_readnoise = np.median(np.array(readnoises))

    print ('gain ccd %s :'%nhdu, med_gain)
    print ('readout noise ccd %s :'%nhdu, med_readnoise)

    #return med_gain * u.electron / u.adu, med_readnoise * u.electron
    return med_gain, med_readnoise



### MASTERBIAS ###

def make_masterbias (rawdir, intdir, masterdir, images, nhdu, gain, ron, poly_model, correct, trimsection) :
    
    # creates the lists of bias frames
    match = (images.summary['object'] == 'BIAS')
    biases = images.summary['file'][match]
    
    print ('hdu %s' %nhdu)
    biaslist = []
    for bias in biases :
        print ('Reducing bias %s ...' %bias)
        rawbias = os.path.join (rawdir, bias)
        ccd = ccdproc.CCDData.read(rawbias, hdu=nhdu, unit=u.adu)
        
        #ccd = ccdproc.subtract_overscan(ccd, overscan_axis=1, fits_section='[2095:2142,1:4128]', model=poly_model)
        #ccd = ccdproc.subtract_overscan(ccd, fits_section='[2095:2142,1:4128]', median=True)
        #ccd = ccdproc.subtract_overscan(ccd, fits_section='[2095:2142,1:4128]')
        ccd = ccdproc.subtract_overscan(ccd, fits_section='[2098:2139,1:4128]')
        #ccd = ccdproc.subtract_overscan(ccd, fits_section='[5:37,1:4128]')

        if correct :
            ccd = ccdproc.trim_image(ccd, fits_section=trimsection)
            
        biaslist.append(ccd)
    
    print ('Creating masterbias%s ...' %nhdu)
    combiner = ccdproc.Combiner(biaslist)
    masterbias = combiner.median_combine()
    
    if correct :
        masterbias.write(os.path.join(masterdir,'masterbias%s_corr.fits'%nhdu), overwrite=True)
    else: 
        masterbias.write(os.path.join(masterdir,'masterbias%s.fits'%nhdu), overwrite=True)
    
    return masterbias


### MASTERFLAT ###

'''                                                                                                                                                        
An uncertainty frame is created for each flat which is then gain corrected, overscan and bias subtracted, normalized to the median                     
'''

def make_flatlist (rawdir, intdir, images, filtname, nhdu, gain, ron, poly_model, masterbias, correct, trimsection) :

    # creates the lists of flatfield frames
    match = (images.summary['object'] == 'FLAT,SKY') & (images.summary['eso ins filt1 name'] == filtname)
    flats = images.summary['file'][match]
    
    print ('hdu %s' %nhdu)
    flatlist = []
    for flat in flats :
        print ('Reducing flat %s ...' %flat)
        rawflat = os.path.join (rawdir, flat)
        ccd = ccdproc.CCDData.read(rawflat, hdu=nhdu, unit=u.adu)
        
        #ccd = ccdproc.subtract_overscan(ccd, overscan_axis=1, fits_section='[2095:2142,1:4128]', model=poly_model)
        #ccd = ccdproc.subtract_overscan(ccd, fits_section='[2095:2142,1:4128]', median=True)
        #ccd = ccdproc.subtract_overscan(ccd, fits_section='[2095:2142,1:4128]')
        ccd = ccdproc.subtract_overscan(ccd, fits_section='[2098:2139,1:4128]')
        #ccd = ccdproc.subtract_overscan(ccd, fits_section='[5:37,1:4128]')
        
        if correct :
            ccd = ccdproc.trim_image(ccd, fits_section=trimsection)
            
        ccd = ccdproc.subtract_bias(ccd, masterbias)

        #ccd = ccdproc.create_deviation(ccd, gain=gain, readnoise=ron)
        ccd = ccdproc.create_deviation(ccd, gain=gain, readnoise=ron, disregard_nan=True)
        ccd = ccdproc.gain_correct(ccd, gain)
        
        flatlist.append(ccd)
            
    return flatlist
    

def flat_norm_factors (flatlists) :
	
    # create a matrix with the median of CCDData objects for each flat frame and CCD
    arr_median = np.zeros ((8,len(flatlists[0])))
    
    print ('Computing medians of each CCD (for each flatfield frame, be patient!) ...')
    
    for i in range(8) :
        print ('CCD %s' %(i+1))
        for j,iccd in enumerate(flatlists[i]) :
            arr_median[i,j] = np.nanmedian(iccd)
            print ('median of flat %s : %s' %(j+1, arr_median[i,j]))
    
    median_of_medians = [np.nanmedian(arr_median[:,col]) for col in range(arr_median.shape[1])]
    inv_of_median_of_medians = [1/np.nanmedian(arr_median[:,col]) for col in range(arr_median.shape[1])]
    print ('median fo medians: ', median_of_medians)
    print ('(median fo medians)^-1: ', inv_of_median_of_medians)
    return inv_of_median_of_medians



def make_masterflat (masterdir, norm_factors, flatlist, filtname, nhdu, correct, egain=None) :

    scaling_factors = norm_factors    # list of factors to normalize the flats (same length of flatlist)

    #combiner.scaling = scaling_factors
    flatlist = [flat.multiply(norm_factors[i]) for i,flat in enumerate(flatlist)]
    combiner = ccdproc.Combiner(flatlist)
    masterflat = combiner.median_combine()
    #print ('+++++++')
    #print (np.median(masterflat3.data))
    #print (np.median(masterflat3.uncertainty.array))
    #print ('+++++++')
    #if correct :
    #    masterflat_corr = masterflat.multiply(egain) 
    
    if filtname == 'BB#Rc/162_ESO844' :
        if correct :
            print ('Correcting masterflat_r%s ...' %nhdu)
            #masterflat_corr.write(os.path.join(masterdir,'masterflat_r%s_corr.fits'%nhdu), overwrite=True)
            masterflat.write(os.path.join(masterdir,'masterflat_r%s_corr.fits'%nhdu), overwrite=True)
        else :
            print ('Creating masterflat_r%s ...' %nhdu)
            masterflat.write(os.path.join(masterdir,'masterflat_r%s.fits'%nhdu), overwrite=True)
    elif filtname == 'NB#Halpha/7_ESO856' :
        if correct :
            print ('Correcting masterflat_h%s ...' %nhdu)
            #masterflat_corr.write(os.path.join(masterdir,'masterflat_h%s_corr.fits'%nhdu), overwrite=True)
            masterflat.write(os.path.join(masterdir,'masterflat_h%s_corr.fits'%nhdu), overwrite=True)
        else :
            print ('Creating masterflat_h%s ...' %nhdu)
            masterflat.write(os.path.join(masterdir,'masterflat_h%s.fits'%nhdu), overwrite=True)
    else :
        sys.exit('Error: filter not defined!')

    return masterflat 
        

### BAD PIXEL MASK ###

def bad_pixel_mask (masterdir, masterflat, filtname, nhdu, config) :
    
    print ('Creating bad pixel mask of CCD %s (it might take some time!) ...' %nhdu)
    '''
    bpm = ccdproc.ccdmask (masterflat, findbadcolumns=True, ncmed=12, nlmed=12, ncsig=25, nlsig=25, lsigma=9, hsigma=9) # WFI 1.0--2.3
    ncmed=7, nlmed=7, ncsig=12, nlsig=12 are the default values
    byblocks=True and ngood>5 do not work for coulumns. findbadcolumns=True is also not affecting the majority of columns.
    '''
    findbadcolumns = config.getboolean('ccdmask', 'findbadcolumns')
    ncmed = config.getint('ccdmask', 'ncmed')
    nlmed = config.getint('ccdmask', 'nlmed')
    ncsig = config.getint('ccdmask', 'ncsig')
    nlsig = config.getint('ccdmask', 'nlsig')
    lsigma = config.getfloat('ccdmask', 'lsigma')
    hsigma = config.getfloat('ccdmask', 'hsigma')
    bpm = ccdproc.ccdmask (masterflat, findbadcolumns=findbadcolumns, ncmed=ncmed, nlmed=nlmed, ncsig=ncsig, nlsig=nlsig, lsigma=lsigma, hsigma=hsigma)
    # update with bad pixel columns not individuated by ccdmask routine
    nccd, X, Y = np.loadtxt('bad_pixel_columns.txt', dtype='int', unpack=True)
    for i in range(len(X)) :
        if nccd[i]==nhdu :
            bpm[:Y[i],X[i]-1] = 1
    if filtname == 'BB#Rc/162_ESO844' :
        np.save (os.path.join(masterdir,'bpm_r%s.npy'%nhdu), bpm)
    elif filtname == 'NB#Halpha/7_ESO856' :
        np.save (os.path.join(masterdir,'bpm_h%s.npy'%nhdu), bpm)
    else :
        sys.exit('Error: filter not defined!')
    
    return bpm


### EMPIRICAL GAIN & READOUT NOISE CORRECTION
'''
The first CCD is used as reference, hence the first emprical gain is equal to the computed gain.
Other gains are computed "empirically" by scaling areas of masterflats, adjacent to each other, for each couple of ccds 
(e.g. lower right corner of CCD1 and lower left of CCD2).
'''

def empirical_gain (masterflat1, masterflat2, n2hdu, bpm1, bpm2, egain1) :
    
    if n2hdu <= 4 :
        ccd1 = np.ma.array(masterflat1.data[50:250, 2062:2092] * egain1, mask=bpm1[50:250, 2062:2092])
        ccd2 = np.ma.array(masterflat2.data[50:250, 53:83], mask=bpm2[50:250, 53:83])
    elif n2hdu == 8: # This connects lower right corner of CCD1 to upper right corner of CCD8
        ccd1 = np.ma.array(masterflat1.data[50:80, 1892:2092] * egain1, mask=bpm1[50:80, 1892:2092])
        ccd2 = np.ma.array(masterflat2.data[3878:4078, 1892:2092], mask=bpm2[3878:4078, 1892:2092])
    else :  # This connects the lower to upper left corners from CCD2 to CCD7 and CCD3 to CCD6
        ccd1 = np.ma.array(masterflat1.data[50:80, 53:253] * egain1, mask=bpm1[50:80, 53:253])
        ccd2 = np.ma.array(masterflat2.data[4048:4078, 53:253], mask=bpm2[4048:4078, 53:253])

        # ccd1 = np.ma.array(masterflat1.data[3878:4078, 53:83] * egain1, mask=bpm1[3878:4078, 53:83])
        # ccd2 = np.ma.array(masterflat2.data[3878:4078, 2062:2092], mask=bpm2[3878:4078, 2062:2092])
    
    emp_gain = np.ma.median(ccd1) / np.ma.median(ccd2)
    print(emp_gain)
    if not np.isfinite(emp_gain):
        emp_gain=1.
    print(emp_gain)
    return emp_gain

def correct_gain (args, masterdir, masterflat, egain, gain, filtname, nhdu, trimsection) :
    
    outflat_r_corr = os.path.join (masterdir, 'masterflat_r%s_corr.fits' %nhdu)
    outflat_h_corr = os.path.join (masterdir, 'masterflat_h%s_corr.fits' %nhdu)   
    
    if (args.master) + (not os.path.exists(outflat_r_corr)) + (not os.path.exists(outflat_h_corr)) :

        masterflat_corr = masterflat.multiply(egain) 
        masterflat_corr = ccdproc.trim_image(masterflat_corr, fits_section=trimsection)
        
        if filtname == 'BB#Rc/162_ESO844' :
            print ('Correcting gain in masterflat_r%s ...' %nhdu)
            masterflat_corr.write(outflat_r_corr, overwrite=True)
        elif filtname == 'NB#Halpha/7_ESO856' :
            print ('Correcting gain in masterflat_h%s ...' %nhdu)
            masterflat_corr.write(outflat_h_corr, overwrite=True)
    
    return egain * gain


### FRAMES REDUCTION ###

def basic_reduction (args, config, rawdir, intdir, masterdir, outdir, images, filtname, nhdu, gain, ron, poly_model, masterbias, masterflat, bpm, chipnum, trimsection) :
    
    # WFI camera characteristic sizes
    pixscale = 0.238/3600. # * u.degree / u.pixel
    xarcsec = 486.948 # x axis dimension in arcsec
    yarcsec = 975.324 # y axis dimension in arcsec
    xgap = 22.848 # gap between ccd in x direction
    ygap = 14.28 # gap between ccd in y direction
    dra = (xarcsec + xgap) / 3600. # deg
    ddec = (yarcsec + ygap)/ 3600. # deg 

    objs = [obj for obj in images.values('object', unique=True) if obj not in ['FLAT,SKY','BIAS','DARK','FLAT,SCREEN','FLUX,STD']]
    for obj in images.values('eso obs targ name', unique=True) :
        if (obj != 'NONE') * (obj not in objs) :
            objs.append(obj)
    sublist = []
    if args.object != [] :
        if set(args.object).issubset(objs) :
            objs = args.object
            #for obj in objs :
            #    print (obj)
        elif set(args.object).issubset(images.summary['file']) :
            sublist = args.object
            #for img in sublist :
            #    print (img)
        else :
            print ('\n')
            sys.exit('Not a valid list of objects! The input images or objects are not found in the observation date')


    def group_runs(li,tolerance=1):
        out = []
        last = li[0]
        for x in li:
            if x-last > tolerance:
                yield out
                out = []
            out.append(x)
            last = x
        yield out

    matches = [((images.summary['object'] == obj) + (images.summary['eso obs targ name'] == obj)) * (images.summary['eso ins filt1 name'] == filtname) for obj in objs]
    
    frameslists = []
    
    for match in matches :
        # order the alphanumeric list
        l = list(images.summary['file'][match])
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        l_sorted = sorted(l, key=alphanum_key)
        # append the ordered list
        frameslists.append(l_sorted)

    print ('hdu %s' %nhdu)
    
    reducedlist = []
    
    for j, frames in enumerate(frameslists) :
        
        for i, img in enumerate(frames) :
            
            if sublist != [] :
                if img not in sublist :
                    continue
            
            print ('Reducing frame %s ...' %img)
            
            rawimg = os.path.join (rawdir, img)
            
            # loading the frame
            ccd = ccdproc.CCDData.read(rawimg, hdu=nhdu, unit=u.adu)
            
            # update the header
            header = fits.open(rawimg)[0].header
            
            if header['OBJECT'] != 'FLUX,STD' :
                ccd.meta['OBJECT'] = header['OBJECT']
                
            else :
                ccd.meta['OBJECT'] = header['ESO OBS TARG NAME']
               
            ccd.meta['FILTNAME'] = header['ESO INS FILT1 NAME']
            ccd.meta['CHIPINDE'] = chipnum
            
            if nhdu <= 4 :
                ccd.meta['CHIPX'] = nhdu
                ccd.meta['CHIPY'] = 2
            else :
                ccd.meta['CHIPX'] = chipnum - 4
                ccd.meta['CHIPY'] = 1
            
            ccd.meta['TRIMSEC'] = trimsection   
            
            ### ROUGH WCS SOLUTION ###
            '''
            We first inizialize galaxy coordinates, tangent point (galaxy centre), ccd where the galaxy is centered
            and we then create a first wcs solution 
            '''
            objectname = ccd.meta['OBJECT']
            obj_txt = '%s_%s_%s.txt' %(objectname, filtname[3:5], args.obsdate)
            if not os.path.exists(os.path.join('objects_txt', obj_txt)) :
                continue
            obj_ccd, ra, dec, crpix1, crpix2 = load_object (os.path.join('objects_txt', obj_txt))
            if config.getboolean('CCD reduction', 'pixel_shifts') :
                for i in range(len(crpix1)) :
                    crpix1[i] = crpix1[i] - 48
                    crpix2[i] = crpix2[i] - 30

            # defining X & Y positions of the CCDs 
            pos = lambda x: {5:[4,1],6:[3,1],7:[2,1],8:[1,1]}.get(x,[x,2])
            X = pos(obj_ccd)[0]
            Y = pos(obj_ccd)[1]
            print ('\tFinding a rough WCS solution ...')
            #hdu = first_wcs (rawdir, img, ccd.meta, X, Y, pixscale, ra, dec, dra, ddec, crpix1[i], crpix2[i])    #, objectname, filtname, obj_ccd)
            hdr = first_wcs (ccd.meta, X, Y, pixscale, ra, dec, dra, ddec, crpix1[i], crpix2[i])    #, objectname, filtname, obj_ccd)
            
            # create wcs
            wcs = WCS(hdr)
            
            # reduction
            ccd = ccdproc.CCDData(ccd.data, mask=bpm, wcs=wcs, meta=ccd.meta, unit=u.adu)     # added mask metadata to the CCDData object
            
            print ('\tOverscan subtraction ...')
            #ccd = ccdproc.subtract_overscan(ccd, overscan_axis=1, fits_section=config.get('Image sections', 'overscan'), model=poly_model)
            #ccd = ccdproc.subtract_overscan(ccd, fits_section='[2095:2142,1:4128]', median=True)
            #ccd = ccdproc.subtract_overscan(ccd, fits_section='[2095:2142,1:4128]')
            ccd = ccdproc.subtract_overscan(ccd, fits_section='[2098:2139,1:4128]')
            #ccd = ccdproc.subtract_overscan(ccd, fits_section='[5:37,1:4128]')
            if args.intermediate :
                ccd.write((os.path.join(intdir,'o%s' %img)).replace('.fits','_%s.fits'%nhdu), overwrite=True)
            
            print ('\tTrimming the image ...')
            ccd = ccdproc.trim_image(ccd, fits_section=ccd.meta['TRIMSEC'])
            if args.intermediate :
                ccd.write((os.path.join(intdir,'to%s' %img)).replace('.fits','_%s.fits'%nhdu), overwrite=True)
            
            print ('\tBias subtraction ...')
            #gain_corrected_masterbias = ccdproc.gain_correct(masterbias, gain)
            #ccd = ccdproc.subtract_bias(ccd, gain_corrected_masterbias)
            ccd = ccdproc.subtract_bias(ccd, masterbias)
            if args.intermediate :
                ccd.write((os.path.join(intdir,'bto%s' %img)).replace('.fits','_%s.fits'%nhdu), overwrite=True)

            print ('\tCreating error frame ...')
            #ccd = ccdproc.create_deviation(ccd, gain=gain, readnoise=ron)
            ccd = ccdproc.create_deviation(ccd, gain=gain, readnoise=ron, disregard_nan=True)
            
            print ('\tGain correction ...')
            ccd = ccdproc.gain_correct(ccd, gain)
            if args.intermediate :
                ccd.write((os.path.join(intdir,'gbto%s' %img)).replace('.fits','_%s.fits'%nhdu), overwrite=True)

            print ('\tRemoving cosmic rays ...')
            '''
            Lacosmic technique, van Dokkum (2001) as implemented by McCully (2014).
            Note 1 : gain_apply argument was not necessary in ccdproc 1.2 since the gain correction is made
            by default with a gain=1.0, but in version 2.1 the units are automatically converted in electrons.
            Note 2: gain_apply=False should avoid unit conversion but only if gain!=1.0 that seems to be a bug.
            Note on June, 4th 2023: I stick to the default gain=1.0 and gain_apply=True because the gain correction
            is now done before cosmic ray removal.
            Note on June, 27th 2023: gain correction done after cosmic ray removal.
            Note on July 12th 2023: default gain=1.0 and gain_apply=True used and cosmic ray removal done after gain correction
            '''
            
            ccd = ccdproc.cosmicray_lacosmic(ccd, gain=1.0, sigclip=6.5, sigfrac=0.2, objlim=20.0, niter=4, gain_apply=False)  
            if args.intermediate :
                ccd.write((os.path.join(intdir,'crgbto%s' %img)).replace('.fits','_%s.fits'%nhdu), overwrite=True)

            # final image corrected
            print ('\tFlat field correction ...')
            #ccd = ccdproc.flat_correct(ccd, masterflat)
            ccd_fc = ccd.divide(masterflat)
            ccd_fc.unit = u.electron    # necessary for ccdproc 2.1
            ccd_fc.uncertainty.unit = u.electron    # necessary for ccdproc 2.1
            ccd_fc = ccdproc.CCDData(ccd_fc.data, uncertainty=ccd_fc.uncertainty, mask=ccd.mask, wcs=ccd.wcs, meta=ccd.meta, unit=u.electron)
            if args.intermediate :
                ccd_fc.write((os.path.join(intdir,'fcrgbto%s'%img)).replace('.fits','_%s.fits'%nhdu), overwrite=True)
            
            # normalized to electron per second
            print ('\tNormalization to electron/second ...')
            ccd_norm = ccd_fc.divide(ccd_fc.meta['exptime'])
            ccd_norm.unit = u.electron    # necessary for ccdproc 2.1
            ccd_norm.uncertainty.unit = u.electron    # necessary for ccdproc 2.1
            '''
            equivalent to
            ccd_norm.data = ccd_fc.data / ccd_fc.meta['exptime']
            ccd_norm.uncertainty = ccd_fc.uncertainty.array / ccd_fc.meta['exptime']
            '''
            ccd_norm = ccdproc.CCDData(ccd_norm.data, uncertainty=ccd_norm.uncertainty, mask=ccd_fc.mask, wcs=ccd_fc.wcs, meta=ccd_fc.meta, unit=u.electron)
            #ccd.write((os.path.join(masterdir,'nfbtocrg%s'%img)).replace('.fits','_%s.fits'%nhdu), overwrite=True)
            
            # write the file with rough wcs
            ccd_norm.write(os.path.join(outdir, 'rwcs%s_%s_%s_frame%s.fits' %(objectname, filtname[3:5], ccd.meta['CHIPINDE'], i+1)), overwrite=True)
            print ('rwcs%s_%s_%s_frame%s.fits created!\n' %(objectname, filtname[3:5], ccd.meta['CHIPINDE'], i+1))

            reducedlist.append(ccd_norm)

    return reducedlist
    

def WFIreduce (args, config) :
    
    # initializing the working directories
    obsdir = os.path.join (args.pathtodata, args.obsdate)
    rawdir = os.path.join (obsdir, 'rawdata')
    intdir = os.path.join (obsdir, 'processed', 'intermediate')
    masterdir = os.path.join (obsdir, 'processed', 'masterframes')
    outdir = os.path.join (obsdir, 'processed', 'output')
    try:
        os.makedirs(intdir)
    except OSError:
        pass
    try:
        os.makedirs(masterdir)
    except OSError:
        pass
    try:
        os.makedirs(outdir)
    except OSError:
        pass
	
	# list of fits file created with ImageFileCollection
    images = make_list(rawdir)
    
    # gain and readout noise computations or initialization
    outgain = os.path.join(masterdir, 'outgain.json')
    outron = os.path.join(masterdir, 'outron.json')
    
    if (args.janesick) + (not os.path.exists(outgain)) + (not os.path.exists(outron)) :
        # gain and readout noise calculated with Janesick's method
        gain = {}
        ron = {}
        for i in range(8) :
            gain['gain%s'%(i+1)], ron['ron%s'%(i+1)] = janesick (config, rawdir, images, i+1) 
        with open(outgain, 'w') as outfile:
            json.dump(gain, outfile, sort_keys=True)
        with open(outron, 'w') as outfile:
            json.dump(ron, outfile, sort_keys=True)
    
    with open(outgain) as data_file:
        gain = json.load(data_file)
    with open(outron) as data_file:
        ron = json.load(data_file)
    for i in range(8) :
        gain['gain%s'%(i+1)] = gain['gain%s'%(i+1)] * u.electron / u.adu
        print ('gain ccd %s :'%(i+1), gain['gain%s'%(i+1)])
        ron['ron%s'%(i+1)] = ron['ron%s'%(i+1)] * u.electron
        print ('readout noise ccd %s :'%(i+1), ron['ron%s'%(i+1)])

    correct = False    # boolean for switching between gain and corrected gain

    poly_model = models.Legendre1D(config.getint('Image sections', 'polynomial_degree'))    # model to fit the overscan columns (default polynomial degree of 14)

    chipindex = lambda x: {5:8,6:7,7:6,8:5}.get(x,x)    # transformation from hdu number to chip index

    trimsection = np.loadtxt ('TrimSections.txt', dtype='str', usecols=(1,))    # load trim sections
    
    # load/create masterbias (for each CCD)
    masterbias = {}
    for i in range(8) :
        outmaster = os.path.join(masterdir, 'masterbias%s.fits' %(i+1))
        if (args.master) + (not os.path.exists(outmaster)) :
            masterbias['masterbias%s'%(i+1)] = make_masterbias(rawdir, intdir, masterdir, images, i+1, gain['gain%s'%(i+1)], ron['ron%s'%(i+1)], poly_model, correct, trimsection[chipindex(i+1)-1])
        else :
            masterbias['masterbias%s'%(i+1)] = ccdproc.CCDData.read(outmaster, unit=u.adu)

    # load masterflats or create list of flats (for each CCD and for each filter)
    flatlist_r = []
    flatlist_h = []
    masterflat_r = {}
    masterflat_h = {}
    for i in range(8) :
        outflat_r = os.path.join (masterdir, 'masterflat_r%s.fits' %(i+1))
        if (args.master) + (not os.path.exists(outflat_r)) :
            flatlist_r.append(make_flatlist(rawdir, intdir, images, 'BB#Rc/162_ESO844', i+1, gain['gain%s'%(i+1)], ron['ron%s'%(i+1)], poly_model, masterbias['masterbias%s'%(i+1)], correct, trimsection[chipindex(i+1)-1]))
        else :
            masterflat_r['masterflat_r%s'%(i+1)] = ccdproc.CCDData.read(outflat_r, unit=u.electron)
        outflat_h = os.path.join (masterdir, 'masterflat_h%s.fits' %(i+1))
        if (args.master) + (not os.path.exists(outflat_h)) :
            flatlist_h.append(make_flatlist(rawdir, intdir, images, 'NB#Halpha/7_ESO856', i+1, gain['gain%s'%(i+1)], ron['ron%s'%(i+1)], poly_model, masterbias['masterbias%s'%(i+1)], correct, trimsection[chipindex(i+1)-1]))
        else :
            masterflat_h['masterflat_h%s'%(i+1)] = ccdproc.CCDData.read(outflat_h, unit=u.electron)
	
    # create normalisation factors with the median of all CCDs of a given flat frame
	# --> one factor each frame instead of one factor each CCD of each frame !!!
    if flatlist_r != [] :
        norm_factors_r = flat_norm_factors(flatlist_r)
    if flatlist_h != [] :
        norm_factors_h = flat_norm_factors(flatlist_h)
	
    # create masterflats (for each CCD and for each filter)
    if masterflat_r == {} :
        for i in range(8) :
	        masterflat_r['masterflat_r%s'%(i+1)] = make_masterflat (masterdir, norm_factors_r, flatlist_r[i], 'BB#Rc/162_ESO844', i+1, correct)
    if masterflat_h == {} :
        for i in range(8) :
            masterflat_h['masterflat_h%s'%(i+1)] = make_masterflat (masterdir, norm_factors_h, flatlist_h[i], 'NB#Halpha/7_ESO856', i+1, correct)
	
    # bad pixel mask with masterflat frames (we don't have high- and low-count dome flats)
    bpm_r = {}
    bpm_h = {}    
    for i in range(8) :
        outbpm_r = os.path.join (masterdir, 'bpm_r%s.npy' %(i+1))
        if (args.badpixel) + (not os.path.exists(outbpm_r)) :
            bpm_r['bpm_r%s'%(i+1)] = bad_pixel_mask (masterdir, masterflat_r['masterflat_r%s'%(i+1)], 'BB#Rc/162_ESO844', i+1, config)
        else :
            bpm_r['bpm_r%s'%(i+1)] = np.load (outbpm_r)
        outbpm_h = os.path.join (masterdir, 'bpm_h%s.npy' %(i+1))
        if (args.badpixel) + (not os.path.exists(outbpm_h)) :
            bpm_h['bpm_h%s'%(i+1)] = bad_pixel_mask (masterdir, masterflat_h['masterflat_h%s'%(i+1)], 'NB#Halpha/7_ESO856', i+1, config)
        else :
            bpm_h['bpm_h%s'%(i+1)] = np.load (outbpm_h)

    # empirical correction of gain and readout noise
    correct = True
    
    # find the empirical gain
    outegain = os.path.join(masterdir, 'outegain.json')
    outnewgain = os.path.join(masterdir, 'outnewgain.json')
    outnewron = os.path.join(masterdir, 'outnewron.json')
    if (args.empirical) + (not os.path.exists(outegain)) + (not os.path.exists(outnewgain)) + (not os.path.exists(outnewron)) :
        egain = {}
        egain['egain1'] = 1. 
        for i in range(4) :
            print(i)
            egain['egain%s'%(i+2)] = empirical_gain (masterflat_r['masterflat_r%s'%(i+1)], masterflat_r['masterflat_r%s'%(i+2)], i+2, 
                bpm_r['bpm_r%s'%(i+1)], bpm_r['bpm_r%s'%(i+2)], egain['egain%s'%(i+1)])
        for i in range(4,7) :
            print(i)
            egain['egain%s'%(i+2)] = empirical_gain (masterflat_r['masterflat_r%s'%(7-i)], masterflat_r['masterflat_r%s'%(i+2)], i+2, 
                bpm_r['bpm_r%s'%(7-i)], bpm_r['bpm_r%s'%(i+2)], egain['egain%s'%(7-i)])
        with open(outegain, 'w') as outfile:
            json.dump(egain, outfile, sort_keys=True)
        # correct for the new empirical gain 
        with open(outgain) as data_file:
            gain = json.load(data_file)
        newgain = {}
        for i in range(8) :
            newgain['newgain%s'%(i+1)] = correct_gain (args, masterdir, masterflat_r['masterflat_r%s'%(i+1)], egain['egain%s'%(i+1)], gain['gain%s'%(i+1)], 'BB#Rc/162_ESO844', i+1, trimsection[chipindex(i+1)-1])
            newgain['newgain%s'%(i+1)] = correct_gain (args, masterdir, masterflat_h['masterflat_h%s'%(i+1)], egain['egain%s'%(i+1)], gain['gain%s'%(i+1)], 'NB#Halpha/7_ESO856', i+1, trimsection[chipindex(i+1)-1])
            #newgain['newgain%s'%(i+1)] = egain['egain%s'%(i+1)] * gain['gain%s'%(i+1)]
            print ('Corrected gain for CCD %s :'%(i+1), newgain['newgain%s'%(i+1)])
        with open(outnewgain, 'w') as outfile:
            json.dump(newgain, outfile, sort_keys=True)
        # find the empirical readout noise
        with open(outron) as data_file:
            ron = json.load(data_file)
        newron ={}
        for i in range(8) :
            newron['newron%s'%(i+1)] = egain['egain%s'%(i+1)] * ron['ron%s'%(i+1)]
            print ('Corrected readout noise for CCD %s :'%(i+1), newron['newron%s'%(i+1)])
        with open(outnewron, 'w') as outfile:
            json.dump(newron, outfile, sort_keys=True)
    
    with open(outegain) as data_file:
        egain = json.load(data_file)
    with open(outnewgain) as data_file:
        newgain = json.load(data_file)
    with open(outnewron) as data_file:
        newron = json.load(data_file)
    for i in range(8) :
        print ('CCD %s'%(i+1))
        print ('Empirical gain :', egain['egain%s'%(i+1)])
        newgain['newgain%s'%(i+1)] = newgain['newgain%s'%(i+1)] * u.electron / u.adu
        print ('Corrected gain :', newgain['newgain%s'%(i+1)])
        newron['newron%s'%(i+1)] = newron['newron%s'%(i+1)] * u.electron
        print ('Corrected readout noise :', newron['newron%s'%(i+1)])
        outflat_r_corr = os.path.join (masterdir, 'masterflat_r%s_corr.fits' %(i+1))
        print ('Loading %s ...' %os.path.basename(outflat_r_corr))
        masterflat_r['masterflat_r%s'%(i+1)] = ccdproc.CCDData.read(outflat_r_corr, unit=u.electron)
        outflat_h_corr = os.path.join (masterdir, 'masterflat_h%s_corr.fits' %(i+1))
        print ('Loading %s ...' %os.path.basename(outflat_h_corr))
        masterflat_h['masterflat_h%s'%(i+1)] = ccdproc.CCDData.read(outflat_h_corr, unit=u.electron)        

    # correcting masterbias
    for i in range(8) :
        outmaster_corr = os.path.join(masterdir, 'masterbias%s_corr.fits' %(i+1))
        if (args.master) + (not os.path.exists(outmaster_corr)) :
            masterbias['masterbias%s'%(i+1)] = make_masterbias(rawdir, intdir, masterdir, images, i+1, newgain['newgain%s'%(i+1)], newron['newron%s'%(i+1)], poly_model, correct, trimsection[chipindex(i+1)-1])
        else :
            masterbias['masterbias%s'%(i+1)] = ccdproc.CCDData.read(outmaster_corr, unit=u.adu)   

    # ccd basic reduction (for each CCD)
    reducedlist_r = []
    reducedlist_h = []
    if args.detrend :
        for i in range(8) :
            reducedlist_r.append (basic_reduction (args, config, rawdir, intdir, masterdir, outdir, images, 'BB#Rc/162_ESO844', i+1, newgain['newgain%s'%(i+1)], newron['newron%s'%(i+1)], poly_model, masterbias['masterbias%s'%(i+1)], masterflat_r['masterflat_r%s'%(i+1)], bpm_r['bpm_r%s'%(i+1)], chipindex(i+1), trimsection[chipindex(i+1)-1]))
            reducedlist_h.append (basic_reduction (args, config, rawdir, intdir, masterdir, outdir, images, 'NB#Halpha/7_ESO856', i+1, newgain['newgain%s'%(i+1)], newron['newron%s'%(i+1)], poly_model, masterbias['masterbias%s'%(i+1)], masterflat_h['masterflat_h%s'%(i+1)], bpm_h['bpm_h%s'%(i+1)], chipindex(i+1), trimsection[chipindex(i+1)-1]))
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prefix_chars='-+')

    parser.add_argument('pathtodata', help='path to data and/or to observation dates folders')
    parser.add_argument('obsdate', help='observation date containing data to reduce')
    parser.add_argument('-o', '--object', action="append", default=[], help='specific target to reduce')
    parser.add_argument('-j', '--janesick', action="store_true", default=False, help='compute the gain with janesick algorithm')
    parser.add_argument('-m', '--master', action="store_true", default=False, help='create master frames for detrending')
    parser.add_argument('-e', '--empirical', action="store_true", default=False, help='compute the empirical gain from the masterflats')
    parser.add_argument('-b', '--badpixel', action="store_true", default=False, help='force creating a bad pixel mask')
    parser.add_argument('-i', '--intermediate', action="store_true", default=False, help='save the intermediate products of image detrend')
    parser.add_argument('-d', '--detrend', action="store_true", default=False, help='detrend the images, compute gain and readout noise, write a rough astrometric solution in the headers, create a mask and an error frame for each CCD')
    
    args = parser.parse_args()

    #Read/Write configuration file
    if not os.path.exists('WFIpipeline.config') :
        print ('\n\nConfiguration file not found!\n\nGenerating a new WFIpipeline.config with internal configuration ...\n\n')
        ConfigWrite()
    else :
        print ('\n\nReading WFIpipeline.config file ...\n\n')
    config = ConfigParser()
    config.read('WFIpipeline.config')

    WFIreduce(args, config)
