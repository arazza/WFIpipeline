# -*- coding: utf-8 -*-

import seaborn as sns

class PaperPalette (object) :
    
    def __init__(self) : 
             
        self.cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        self.palette = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=False)
        self.color1 = (self.palette[0][0], self.palette[0][1], self.palette[0][2])
        self.color2 = (self.palette[1][0], self.palette[1][1], self.palette[1][2])
        self.color3 = (self.palette[2][0], self.palette[2][1], self.palette[2][2])
        self.color4 = (self.palette[3][0], self.palette[3][1], self.palette[3][2])
        self.color5 = (self.palette[4][0], self.palette[4][1], self.palette[4][2])
        self.color6 = (self.palette[5][0], self.palette[5][1], self.palette[5][2])