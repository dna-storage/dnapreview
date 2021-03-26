#from numpy import array
#from numpy import mean
from scipy.fftpack import *
from scipy.fftpack import dct,idct
import numpy as np
import unittest
import random

dQ = np.array([ [16, 11, 10, 16, 24, 40, 51, 61],
      [12, 12, 14, 19, 26, 58, 60, 55],
      [14, 13, 16, 24, 40, 57, 69, 56],
      [14, 17, 22, 29, 51, 87, 80, 62],
      [18, 22, 37, 56, 68, 109, 103, 77],
      [24, 35, 55, 64, 81, 104, 113, 92],
      [49, 64, 78, 87, 103, 121, 120, 101],
      [72, 92, 95, 98, 112, 100, 103, 99] ])

def _alpha(u):
    if u==0:
        return 1.0/np.sqrt(2)
    else:
        return 1
        
DCT_tensor = None
DCT_tensor_init = False
def getDCTTerm(u,v):    
    global DCT_tensor_init
    global DCT_tensor
    if DCT_tensor_init:
        return DCT_tensor[u][v]
    else:
        u_p = u
        v_p = v
        DCT_tensor_init = True
        DCT_tensor = np.zeros((8,8,8,8))
        for u in range(0,8):
            for v in range(0,8):
                Xm_row = np.array([ float(i) for i in range(8) ])
                Xm_row = np.cos( (2*Xm_row+1)*u*np.pi/16.0 )
                Xm = [ Xm_row for _ in range(8) ]
                Xm = np.array(Xm)
                Ym_row = np.array([ float(i) for i in range(8) ])
                Ym_row = np.cos( (2*Ym_row+1)*v*np.pi/16.0 )
                Ym = [ Ym_row for _ in range(8) ]
                Ym = np.transpose(np.array(Ym))
                DCT_tensor[u][v] = Xm * Ym * 1.0/4 * _alpha(u)*_alpha(v)
        return getDCTTerm(u_p,v_p)

IDCT_tensor_init = False
IDCT_tensor = None
def getIDCTTerm(x,y):
    global IDCT_tensor
    global IDCT_tensor_init
    if IDCT_tensor_init:
        return IDCT_tensor[y][x]
    else:
        IDCT_tensor_init = True
        IDCT_tensor = np.zeros((8,8,8,8))
        x_p = x
        y_p = y
        m = [ [ i for i in range(8) ] for _ in range(8) ]
        Um = np.array(m)
        Vm = np.transpose(Um)
        alphaVM = [ [1/np.sqrt(2) for i in range(8)] ]
        for i in range(7):
            alphaVM.append ( [1 for _ in range(8)] )
        alphaVM = np.array(alphaVM)
        alphaUM = np.transpose(alphaVM)
        #print "Um = {}".format(Um)
        #print "Vm = {}".format(Vm)
        #print "alphaUm = {}".format(alphaUM)
        #print "alphaVm = {}".format(alphaVM)

        for x in range(8):
            for y in range(8):
                res = np.cos( (2*x+1)*Um*np.pi / 16.0 ) * np.cos( (2*y+1)*Vm*np.pi / 16.0 )                
                IDCT_tensor[y][x] = alphaUM * alphaVM * res / 4.0
        return getIDCTTerm(x_p,y_p)

def sci_idct2(a):
    return idct(idct(a,axis=0,norm='ortho'),axis=1,norm='ortho')

def sci_dct2(a):
    return dct(dct(a,axis=0,norm='ortho'),axis=1,norm='ortho')

def idct2(b,Q=dQ):
    b = np.array(b).astype(float)*Q
    f = sci_idct2(b)
    # f = np.zeros( (8,8) )
    # for y in range(8):
    #     for x in range(8):
    #         f[y][x] = np.sum( getIDCTTerm(x,y) * b ) #f_x_y(b,x,y)
    f = np.rint(f+128)
    f = np.clip(f,0,255)
    #print f
    return f.astype(int)


def dct2(b,Q=dQ):
    b = np.array(b)-128
    G = np.zeros( (8,8) )    
    # for y in range(8):
    #      for x in range(8):
    #          G[y][x] = np.sum(b * getDCTTerm(x,y))
    G = sci_dct2(b)
    G = np.rint(G/Q)
    return G.astype(int)


# Y,X
zig_zag_order =    [ (0,0) ] \
        + [ (i, 1-i) for i in range(2) ] \
        + [ (2-i, i) for i in range(3) ] \
        + [ (i, 3-i) for i in range(4) ] \
        + [ (4-i, i) for i in range(5) ] \
        + [ (i, 5-i) for i in range(6) ] \
        + [ (6-i,i) for i in range(7) ] \
        + [ (i,7-i) for i in range(8) ] \
        + [ (7-i,1+i) for i in range(7) ]\
        + [ (i+2,7-i) for i in range(6) ]\
        + [ (7-i, i+3) for i in range(5) ]\
        + [ (i+4,7-i) for i in range(4) ]\
        + [ (7-i,i+5) for i in range(3) ]\
        + [ (i+6,7-i) for i in range(2) ]\
        + [ (7-i,i+7) for i in range(1) ]

x_zig_zag_order = [ x for (x,y) in zig_zag_order ]
y_zig_zag_order = [ y for (x,y) in zig_zag_order ]

def flatten_by_zig_zag(block):
    block = np.array(block)
    arr = np.zeros( (64) )
    for i,(y,x) in enumerate(zig_zag_order):
        arr[i] = block[y][x]
    return arr

def unflatten_by_zig_zag(block):
    arr = np.zeros( (8,8) )
    for i,(y,x) in enumerate(zig_zag_order):
        arr[y][x] = block[i]
    return arr


class ZigZagTest(unittest.TestCase):
    def test_zig_zag(self):
        print(zig_zag_order)
        correct = np.array( [[  0,   1,   5,   6,  14,  15,  27,  28,],\
                             [  2,   4,   7,  13,  16,  26,  29,  42,],\
                             [  3,   8,  12,  17,  25,  30,  41,  43,],\
                             [  9,  11,  18,  24,  31,  40,  44,  53,],\
                             [ 10,  19,  23,  32,  39,  45,  52,  54,],\
                             [ 20,  22,  33,  38,  46,  51,  55,  60,],\
                             [ 21,  34,  37,  47,  50,  56,  59,  61,],\
                             [ 35,  36,  48,  49,  57,  58,  62,  63,]])

        arr = np.zeros( (8,8) )
        for i,(y,x) in enumerate(zig_zag_order):
            arr[y][x] = i
        arr2 = abs(arr - correct)
        assert np.sum(arr2)==0
        print(arr)
        print(arr2)
        print(flatten_by_zig_zag(arr))
        arr3 = unflatten_by_zig_zag(flatten_by_zig_zag(arr))
        assert np.sum(abs(arr3-arr))==0

def DCT_and_flatten(band,Q=dQ):
    new_band = []
    for b in band:
        new_band.append(flatten_by_zig_zag(dct2(b,Q)))
    return np.array(new_band)

def unflatten_and_IDCT_block(block,Q=dQ):
    #print unflatten_by_zig_zag(block)
    a = np.array(idct2(unflatten_by_zig_zag(block),Q))
    #a = np.clip(a,0,255)
    #a = np.transpose(a)
    #print a
    return a
    
def unflatten_and_IDCT(band,Q=dQ):
    new_band = []
    for b in band:
        new_band.append(idct2(unflatten_by_zig_zag(b),Q))
    return np.array(new_band)


class DCT_and_IDCT_Tests(unittest.TestCase):
    def test_dct_idct(self):
        q = np.ones(64)
        q = q.reshape( (8,8) )
        arr = []
        for i in range(16):
            a = np.array([np.random.randint(256) for _ in range(64)])
            a = a.reshape( (8,8) )
            arr.append(a)
        arr = np.array(arr)
        new = unflatten_and_IDCT(DCT_and_flatten(arr,q),q)
        print(np.sum( arr-new ))
        # This is a hack!
        assert np.sum( arr-new ) < 20
        
    def test_wikipedia_example(self):
        m = np.array([[52,55,61,66,70,61,64,73],   \
                   [63,59,55,90,109,85,69,72],  \
                   [62,59,68,113,144,104,66,73],\
                   [63,58,71,122,154,106,70,69],\
                   [67,61,68,104,126,88,68,70], \
                   [79,65,60,70,77,68,58,75],   \
                   [85,71,64,59,55,61,65,83],   \
                   [87,79,69,68,65,76,78,94]])
        print("original = {}".format(m))
        m_dct = dct2(m)
        print("DCT = {}.".format(m_dct))
        m_idct = idct2(m_dct)
        print("recovered = {}".format(m_idct))
        print("MAE = {}".format(sum( abs(m-m_idct) )/64.0))
        assert np.sum( abs(m-m_idct) )/64.0 < 5.0

    def test_wikipedia_example2(self):
        m = np.array([[-26,-3,-6,2,2,-1,0,0],   \
                   [0,-2,-4,1,1,0,0,0],  \
                   [-3,1,5,-1,-1,0,0,0],\
                   [-3,1,2,-1,0,0,0,0],\
                   [1,0,0,0,0,0,0,0],\
                   [0,0,0,0,0,0,0,0],\
                   [0,0,0,0,0,0,0,0],\
                   [0,0,0,0,0,0,0,0],])
        print("original = {}".format(m))
        m_idct = idct2(m)
        print("IDCT = {}.".format(m_idct))
        m = np.array([[52,55,61,66,70,61,64,73],   \
                   [63,59,55,90,109,85,69,72],  \
                   [62,59,68,113,144,104,66,73],\
                   [63,58,71,122,154,106,70,69],\
                   [67,61,68,104,126,88,68,70], \
                   [79,65,60,70,77,68,58,75],   \
                   [85,71,64,59,55,61,65,83],   \
                   [87,79,69,68,65,76,78,94]])
        assert np.sum( abs(m-m_idct) )/64.0 < 5.0                

        
if __name__ == "__main__":
    m = np.array([[52,55,61,66,70,61,64,73],   \
               [63,59,55,90,109,85,69,72],  \
               [62,59,68,113,144,104,66,73],\
               [63,58,71,122,154,106,70,69],\
               [67,61,68,104,126,88,68,70], \
               [79,65,60,70,77,68,58,75],   \
               [85,71,64,59,55,61,65,83],   \
               [87,79,69,68,65,76,78,94]])
    print("original = {}".format(m))
    m_dct = dct2(m)
    print("DCT = {}.".format(m_dct))
    m_idct = idct2(m_dct)
    print("recovered = {}".format(m_idct))
    print("MAE = {}".format(np.sum( abs(m-m_idct) )/64.0))

    print("*"*30)
    print(zig_zag_order)
    correct = np.array( [[  0,   1,   5,   6,  14,  15,  27,  28,],\
                         [  2,   4,   7,  13,  16,  26,  29,  42,],\
                         [  3,   8,  12,  17,  25,  30,  41,  43,],\
                         [  9,  11,  18,  24,  31,  40,  44,  53,],\
                         [ 10,  19,  23,  32,  39,  45,  52,  54,],\
                        [ 20,  22,  33,  38,  46,  51,  55,  60,],\
                        [ 21,  34,  37,  47,  50,  56,  59,  61,],\
                        [ 35,  36,  48,  49,  57,  58,  62,  63,]])

    arr = np.zeros( (8,8) )
    for i,(x,y) in enumerate(zig_zag_order):
        arr[x][y] = i
    print(arr)
    print(correct)
    arr2 = abs(arr - correct)
    print(arr2)
    assert np.sum(arr2)==0
    print(arr)
    print(arr2)
    arr = unflatten_by_zig_zag([x for x in range(64)])
    print(arr)
    arr2 = abs(arr - correct)
    assert np.sum(arr2)==0
    arr3 = unflatten_by_zig_zag(flatten_by_zig_zag(arr))
    print(flatten_by_zig_zag(arr))
    assert np.sum(arr3-arr)==0


    m = np.array([[-26,-3,-6,2,2,-1,0,0],   \
                  [0,-2,-4,1,1,0,0,0],  \
                  [-3,1,5,-1,-1,0,0,0],\
                  [-3,1,2,-1,0,0,0,0],\
                [1,0,0,0,0,0,0,0],\
                  [0,0,0,0,0,0,0,0],\
                  [0,0,0,0,0,0,0,0],\
                [0,0,0,0,0,0,0,0],])
    print("original = {}".format(m))
    m_idct = idct2(m)
    print("IDCT = {}.".format(m_idct))
    m = np.array([[52,55,61,66,70,61,64,73],   \
                  [63,59,55,90,109,85,69,72],  \
                  [62,59,68,113,144,104,66,73],\
                [63,58,71,122,154,106,70,69],\
                  [67,61,68,104,126,88,68,70], \
                  [79,65,60,70,77,68,58,75],   \
                  [85,71,64,59,55,61,65,83],   \
                  [87,79,69,68,65,76,78,94]])
    assert np.sum( abs(m-m_idct) )/64.0 < 5.0                
