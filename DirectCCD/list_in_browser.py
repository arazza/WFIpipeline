#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 2 2019

@author: arazza
"""

from pathlib import Path
import argparse
from ccdproc import ImageFileCollection

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p', '--pathtofolder', action = 'store', help='path to the folder with fits files to show in browser')
     
    args = parser.parse_args()
    
    images = ImageFileCollection(args.pathtofolder, keywords=['OBJECT','FILTER','EXPTIME'])
    
    images.summary.show_in_browser()
    