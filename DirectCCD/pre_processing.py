#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 5 2024

@author: arazza
"""

import numpy as np

from pathlib import Path
import argparse
import sys
from shutil import copy

from astropy.io import fits

from ccdproc import ImageFileCollection


def make_file (args, filename) :
    
    # Read the paths text file
    
    if args.paths_file != None :
        
        with open(args.paths_file) as file :
            paths = [Path(line.rstrip('\n')) for line in file.readlines()]
        
    else :
        
        paths = [args.pathtogalaxy]
    
    # Create new pre-processing file if not exist
    
    if not Path(filename).exists() :
        
        Path(filename).touch()
    
    # loop over the list of paths
    
    for path in paths :    
        
        print('\nLooking at files in %s ...\n'%path)
        
        images = ImageFileCollection(str(path), keywords=['OBJECT','FILTER'])
        
        for filt in images.values('FILTER', unique=True) : 
            
            for obj in images.values('OBJECT', unique=True) :
                
                if obj==args.old_objname :
                    
                    new_obj = args.new_objname
                    
                else :
                    
                    new_obj = obj
                    
                if filt in args.old_filtnames :
                    
                    pos_old_filt = np.where(np.array(args.old_filtnames)==filt)[0][0]
                    
                    new_filt = args.new_filtnames[pos_old_filt]
            	    
                with Path(filename).open('a') as f :
        	        	
                    f.write('%s %s %s %s %s\n'%(path,obj,new_obj,filt,new_filt))
        
        
def rename (args, filename) :
    
    # Read the paths text file
    
    if args.paths_file != None :
        
        with open(args.paths_file) as file :
            paths = [Path(line.rstrip('\n')) for line in file.readlines()]
        
    else :
        
        paths = [args.pathtogalaxy]
    
    # Path to backup folder
    
    pathtobackup = paths[0].parents[1].joinpath('Obs_run2020_backup')
    
    pathtobackup.mkdir(exist_ok=True)
        
    with Path(filename).open('r') as file :
        
        for line in file.readlines() :

            line = line.rstrip('\n')
            
            line = line.split()
            
            path = line[0]
            
            old_obj = line[1]
            
            new_obj = line[2]
            
            old_filt = line[3]
            
            new_filt = line[4]
            
            if (new_obj!=old_obj) + (new_filt!=old_filt) :
                
                images = ImageFileCollection(str(path), keywords=['file','OBJECT', 'FILTER'])
                
                tbl = images.summary['file','OBJECT','FILTER'][(images.summary['FILTER']==old_filt)&(images.summary['OBJECT']==old_obj)]
            
                if len(tbl['file'])==0 :
                    continue
                
                for f in tbl['file'] :
                    
                    pf = Path(path).joinpath(f)

                    path_to_backup = pathtobackup.joinpath(Path(path).stem)
                    path_to_backup.mkdir(exist_ok=True)

                    pb = path_to_backup.joinpath(f)
                    
                    print('\nSaving the file %s in %s ...\n'%(pf,pb))
                    
                    copy(pf, pb)
                    
                    print('\nUpdating header in %s ...\n'%pf)
                     
                    hdu = fits.open(pf)
                    hdr = hdu[0].header  
                    hdr['OBJECT'] = new_obj
                    hdr['FILTER'] = new_filt
                    hdu[0].header = hdr
                    
                    hdu.writeto(pf, overwrite=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
     
    parser.add_argument('--pathtogalaxy', action='store', help='path to raw frames')
    parser.add_argument('--old_objname', action='store', help='old name of selected galaxy')
    parser.add_argument('--new_objname', action='store', help='new name of selected galaxy')
    parser.add_argument('--old_filtnames', nargs='*', type=str, action='store', default=['r','Ha657','Ha663'], help='old name of the filters')
    parser.add_argument('--new_filtnames', nargs='*', type=str, action='store', default=['r','Ha657','Ha663'], help='new name of the filters')
    parser.add_argument('-p', '--paths_file', action = 'store', help='text file containing paths to each observation date folder')

    parser.add_argument('-m','--make_file', action='store_true', default=False, help='make file where paths, old and new names are stored')
    parser.add_argument('-r','--rename', action='store_true', default=False, help='read file where paths, old and new names are stored')
    
    args = parser.parse_args()
    
    if args.pathtogalaxy == None :
        print('\nPath to raw frames missing!\n')
    
    if args.paths_file == None :
        print('\nText file with paths to data folders missing!\n')

    if (args.pathtogalaxy == None) * (args.paths_file == None) :
        sys.exit('\nBoth the path to raw frames and the text file with the paths to data folders are missing!\nExit the code!\n')
    
    filename = 'pre_processing.txt'

    if args.make_file :
        images = make_file (args, filename)
    
    if args.rename :
        rename (args, filename)

