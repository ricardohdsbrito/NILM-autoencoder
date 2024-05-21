#from shapely.geometry import Polygon
#from shapely.geometry import Point
import numpy as np
import random
from ctypes import cdll
import ctypes
from ctypes import *
import cmath

#Functions for the pq class:

def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)

#comment this out so it can run on macos

'''
def inside_pol():
    dll = ctypes.CDLL('./inside.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.inside
    func.argtypes = [POINTER(c_float), c_int, c_int, POINTER(c_float), c_int, POINTER(c_float)]
    return func

__inside_pol = inside_pol()
'''

def IsInsidePolygon(polygon, pol_x, pol_y, points, k, result):
    pol = polygon.ctypes.data_as(POINTER(c_float))
    pts = points.ctypes.data_as(POINTER(c_float))
    res = result.ctypes.data_as(POINTER(c_float))

    __inside_pol(pol, pol_x, pol_y, pts, k, res)
    

def monte_carlo_area(p, q):

    p_min = np.min(p)
    p_max = np.max(p)
    q_min = np.min(q)
    q_max = np.max(q)

    p_span = p_max - p_min
    q_span = q_max - q_min
    max_points = 10000
    p_rand = [p_min + (random.random() * p_span) for i in range(max_points)]
    q_rand = [q_min + (random.random() * q_span) for i in range(max_points)]
    
    poly = Polygon([(pi, qi) for (pi,qi) in zip(p, q)])
    
    total_points = [(pi, qi) for (pi, qi) in zip(p_rand, q_rand)]
    poly_points = [(pi, qi) for (pi, qi) in total_points if poly.contains(Point(pi,qi))]
    
    rect_area = p_span * q_span
    poly_area = rect_area * len(poly_points)/len(total_points)
    
    return poly_area
 
def colors(n):
    ret = []
    r = int(0.1 * 256)
    g = int(0.3 * 256)
    b = int(0.5 * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b)) 
    return ret


