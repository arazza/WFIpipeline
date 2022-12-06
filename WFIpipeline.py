# coding: utf-8
'''
This code is an adaptation of the original WFI pipeline used to detrend, 
astrometrically register and coadd WFI raw images.

The adaptation is meant to reduce two spectrophotometric standard star fields,
namely LTT3864 and LTT2415.

These parts (no longer necessary) are commented out:
- check if the observation date or data folder are valid (existing) 
- all the calls and argumments connected to the script to re-organize WFI data
'''

import sys
import os
import timeit

import argparse

from configparser import ConfigParser

#from ESO_WFI_reader import reader
from WFIconfig import ConfigWrite
from CCDred_basic import WFIreduce
from astromatic_pipeline import aw_pipeline

# Main

if __name__ == "__main__":
    
    start_time = timeit.default_timer()
    
    parser = argparse.ArgumentParser(prefix_chars='-+')
    
    parser.add_argument('pathtodata', help='path to data and/or to observation dates folders')
    parser.add_argument('obsdate', help='observation date containing data to reduce')
    #parser.add_argument('-r', '--read', action="store_true", default=False, help='read and organize WFI archive data')
    parser.add_argument('-o', '--object', action="append", default=[], help='specific target to reduce')
    parser.add_argument('-j', '--janesick', action="store_true", default=False, help='compute the gain with janesick algorithm')
    parser.add_argument('-m', '--master', action="store_true", default=False, help='create master frames for detrending')
    parser.add_argument('-e', '--empirical', action="store_true", default=False, help='compute the empirical gain from the masterflats')
    parser.add_argument('-b', '--badpixel', action="store_true", default=False, help='force creating a bad pixel mask')
    parser.add_argument('-i', '--intermediate', action="store_true", default=False, help='save the intermediate products of image detrend')
    parser.add_argument('-d', '--detrend', action="store_true", default=False, help='detrend the images, compute gain and readout noise, write a rough astrometric solution in the headers, create a mask and an error frame for each CCD')
    parser.add_argument('-a', '--astrometry', action="store_true", default=False, help='find the astrometric solution')
    parser.add_argument('+a', '--ref_catalog', action="store", default='GAIA-DR1', help='find the astrometric solution with the given reference catalog (default GAIA-DR1)')
    parser.add_argument('+f', '--ref_file', action="store", default='None', help='reference catalog file name')
    parser.add_argument('-c', '--coadd', action="store_true", default=False, help='resample the images to create the mosaics and coadd them together')
    
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.0')
    
    args = parser.parse_args()
    
    print ('\n')
    print ('****************************')
    print ('******* WFI Pipeline *******')
    print ('****************************')
    print ('\n')
    print ('\n')
    print ('\n')
    
    ## Check whether pathtodata and obsdate are correct
    #if not os.path.exists(args.pathtodata) :
    #    sys.exit('%s is not a valid path to WFI data!'%args.pathtodata)
    #
    #if args.read :
    #    obsdates = reader(args.pathtodata)
    #else :
    #    obsdates = []
    #for d in os.listdir(args.pathtodata) :
    #    if os.path.isdir(os.path.join(args.pathtodata, d)) :
    #        if (d not in obsdates) * (d != 'archdata') :
    #            obsdates.append(d)
    #
    #
    #if not args.obsdate in obsdates :
    #    print ('%s is not an existing WFI observation date!'%args.obsdate)
    #    print ('\n')
    #    print ('The following observation dates containing WFI fits files were found or created in %s:\n'%args.pathtodata)
    #    for obs in obsdates :
    #        print (obs)
    #    print ('\n')
    #    newdate = raw_input ('Write the observation date you want to reduce (%s) : ' %args.obsdate)
    #    if newdate in obsdates :
    #        args.obsdate = newdate
    #    else :
    #        sys.exit('%s is not an existing WFI observation date!'%newdate)

    #Read/Write configuration file
    if not os.path.exists('WFIpipeline.config') :
        print ('\n\nConfiguration file not found!\n\nGenerating a new WFIpipeline.config with internal configuration ...\n\n')
        ConfigWrite()
    else :
        print ('\n\nReading WFIpipeline.config file ...\n\n')
    config = ConfigParser()
    config.read('WFIpipeline.config')

    print ('\nReducing data in %s ...\n'%args.obsdate)
    '''
    usage: -j, -m, -b or -d call WFIreduce. Argument -o to list the object to reduce (empty list to reduce all).
    '''
    if (args.janesick) + (args.master) + (args.empirical) + (args.badpixel) + (args.detrend) :
        WFIreduce(args, config)
    '''
    usage: -a to perform only astrometry, -c for stacking only
    '''
    if (args.astrometry) + (args.coadd) :
        list_ref_catalog = ['NONE','FILE','USNO-A2','USNO-B1','GSC-2.3','TYCHO-2','UCAC-4','URAT-1','NOMAD-1','PPMX',
                                'CMC-15','2MASS','DENIS-3','SDSS-R9,IGSL','GAIA-DR1','GAIA-DR2','GAIA-EDR3','ALLWISE']
        if args.ref_catalog == 'FILE' :
            if args.ref_file == 'None' :
                args.ref_file = raw_input('Reference catalog file name : ')
        if args.ref_catalog in list_ref_catalog :
            aw_pipeline(args, config)
        else : 
            print ('Write a valid reference catalog for the astrometry!')
            for c in list_ref_catalog :
                print (c),
            while args.ref_catalog not in list_ref_catalog :
                args.ref_catalog = raw_input('>>> ')
            aw_pipeline(args, config)
        
        

    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)

    print ('\n')
    print ('\n')
    print ('\n')
    print ('*** Pipeline terminated in %d:%02d:%02d ***'%(h, m, s))
    print ('\n')
    print ('\n')
    print ('\n')
