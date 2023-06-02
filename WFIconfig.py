# coding: utf-8

from configparser import ConfigParser


def ConfigWrite () : 
    
    config = ConfigParser()
    
    # General configuration
    config['General'] = {'keep_archive': 'Yes', 'intermediate_products': 'No'}
    
    # Janesick routine for gain and readout noise computation
    config['Janesick'] = {'box_size' : '100'}
    
    # Image sections (trim section, overscan region)
    config['Image sections'] = {'polynomial_degree': '14', 'trimsection': '[49:2094,31:4128]', 'overscan': '[2095:2142,1:4128]'}
    
    # CCD reduction
    config['CCD reduction'] = {'pixel_shifts' : 'No'}

    # CCDMASK
    config['ccdmask'] = {'findbadcolumns': 'True', 'ncmed': '7', 'nlmed': '7', 'ncsig': '12', 'nlsig': '12', 'lsigma': '9', 'hsigma': '9'}
    
    # Image Stacking
    config['Image Stacking'] = {'excl_ccds' : '[5, 6]'}

    # SExtractor
    config['SExtractor'] = {'Execute': 'Yes', 'SourceMinFlux': '20', 'SourceMaxFlux': '7000', 'SourceFluxRadius': '3'}
    
    # SCAMP
    config['Scamp'] = {'NThreads' : '0'}

    # SCWarp
    #config.add_section('SWarp')
    #config.set('SWarp', 'NThreads', 8)


    with open('WFIpipeline.config', 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
   
    ConfigWrite ()