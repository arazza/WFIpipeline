# coding: utf-8

import numpy as np
import os
import sys
import math as m
from glob import glob

import ccdproc
from astropy.time import Time
from shutil import move, copy

from pathlib import Path
from configparser import ConfigParser

def reader (config, pathtodata = '/Volumes/WESO') :
    
    # list of ESO WFI archive data to copy and reduce
    wfifiles = sorted(glob(os.path.join(pathtodata, 'WFI*')))
    for f in wfifiles :
        if f[-1]=='Z' :
            os.system('uncompress %s'%f)
    wfifits = sorted(glob(os.path.join(pathtodata, '*.fits')))

    # directory where to copy the ESO WFI archive already reduced 
    archdir = os.path.join(pathtodata, 'archdata')
    try:
        os.makedirs(archdir)
    except OSError:
        pass

    keys = ['mjd-obs', 'date-obs', 'eso dpr catg', 'eso dpr type', 'origfile', 'arcfile']
    images = ccdproc.ImageFileCollection (pathtodata, keywords=keys)
    
    dt = 3.5/24.    # difference between UTC and CLST=UTC-3h or CLT=UTC-4h
    
    obsdates = []

    if len(wfifits) > 0 :
        while len(images.summary) > 0 :
            # find the minimum mjd date
            mjds = Time(images.summary['mjd-obs'], format='mjd')
            mjd_min = mjds.min().mjd
            # create a directory for an observation date (name set with isot format)
            obsdate = mjds.min().isot[:10]
            obsdates.append(obsdate)
            rawdir = os.path.join(pathtodata, obsdate, 'rawdata')
            try:
                os.makedirs(rawdir)
            except OSError:
                pass
            # filter the collection with the fits from the same observation date
            if ((mjd_min-m.floor(mjd_min))>0.5)*((mjd_min-m.floor(mjd_min))<=0.5+dt) :    # catch the exception of a bias/flat frame taken before 12 CLST!
                mjd_ref = round(mjd_min) - 0.5
            else :
                mjd_ref = round(mjd_min - dt) - 0.5
            match = images.summary['mjd-obs'] <= mjd_ref + 1
            filenames = [f for f in images.summary['file'][match]]
            rawfiles = ccdproc.ImageFileCollection(pathtodata, keywords=keys, filenames=filenames)
            for header, fname in rawfiles.headers(return_fname=True) :
                infile = os.path.join (pathtodata, fname)
                outfile1 = os.path.join (rawdir, header['origfile'])
                outfile2 = os.path.join (archdir, fname)
                if config.getboolean('General', 'keep_archive') :
                    if not os.path.exists (outfile1) :
                        print ('Copying %s in %s ...' %(infile,outfile1))
                        copy (infile, outfile1)
                    if not os.path.exists (outfile2) :
                        print ('Moving %s in %s ...' %(infile,archdir))
                        move (infile, outfile2)
                else :
                    if not os.path.exists (outfile1) :
                        print ('Moving %s in %s ...' %(infile,outfile1))
                        move (infile, outfile1)
            # update the collection with the remaining fits files
            filenames = [f for f in images.summary['file'][~match]]
            if filenames != [] :
                images = ccdproc.ImageFileCollection(pathtodata, keywords=keys, filenames=filenames)
            else :
                images = ccdproc.ImageFileCollection(pathtodata, keywords=keys, filenames='nofiles')
    
    return obsdates


if __name__ == "__main__":
    
    #Read/Write configuration file
    if not Path.exists(Path('WFIpipeline.config')) :
        print ('\n\nConfiguration file not found!\n\nGenerating a new WFIpipeline.config with internal configuration ...\n\n')
        ConfigWrite()
    else :
        print ('\n\nReading WFIpipeline.config file ...\n\n')
    config = ConfigParser()
    config.read('WFIpipeline.config')
    
    obsdates = reader (config)