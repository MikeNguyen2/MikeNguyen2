"""This document contains interactive objects."""
import numpy as np

class Item:
    """The standardclass for the classes below."""

    def __init__(self, name):
        self.name = name
        self.contour = None
        self.pixel_x = None
        self.pixel_y = None  

class Microplate(Item):
    def __init__(self):
        Item.__init__(self, name = 'MTP')
        self.length = 0.085
        self.width = 0.124
        self.height = 0.015
        self.well_diameter = 0.007
        self.well_distance = 0.003
        self.pattern_left = 0.0085
        self.pattern_top = 0.005
        self.dist = 10
        self.radius = 5
        self.canny_limit = 100
        self.lower_limit = 10
        
class Well(Item):
    def __init__(self, contour, position, form='circle'):
        pass

if __name__ == '__main__':
    mtp = Microplate()
    print(mtp.point_x)
