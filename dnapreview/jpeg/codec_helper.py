import numpy as np
import bitarray as ba
from bitarray import bitarray
from dnapreview.jpeg.DCT import unflatten_and_IDCT_block

def nearest(X,C):
    if X % C > 0:
        return (X//C+1)*C
    else:
        return X

    # Todo: replace this hacky interface with import struct
def _get_int(bytes):
    sum = 0
    for b in bytes:
        sum *= 256
        sum += b
    return sum

def _get_int_from_bits(ba):
    #print ba
    val = bitarray( (8 - len(ba) % 8) )
    val.setall(False)
    val = val + ba
    i = _get_int([x for x in val.tobytes()])
    #print "ba={} val={} i={}".format(ba,val,i)
    return i

def byte_unpacker(str,pack_list):
    b = bitarray()
    b.frombytes(str)
    pos = 0
    for i,(l,d) in enumerate(pack_list):
        val = b[pos:pos+l]
        z = bitarray(8 - l % 8)
        z.setall(False)
        val = z + val
        pack_list[i][1] = _get_int([x for x in val.tobytes()])        
        pos += l
    return pack_list

def byte_packer(pack_list):
    """ Expects a tuple of bit length and data value and returns a corresponding """
    """ byte array                                                               """
    b = bitarray()
    for l,d in pack_list:
        b += bitarray("{:0{width}b}".format(d,width=l))
    return b.tobytes()


class ImageComponent:
    def __init__(self, X, Y, init_data=True):
        self.X = X
        self.Y = Y
        if init_data:
            self.data = [ [ None for _ in range(self.X)] for __ in range(self.Y)]

    def set_block(self,x,y,block):
        if x >=0 and x < self.X and y>=0 and y < self.Y:
            self.data[y][x] = block

    def get_block(self,x,y):
        if x >=0 and x < self.X and y>=0 and y < self.Y:
            return self.data[y][x]
        else:
            return None
        
    def flatten(self):
        a = np.array( self.data )
        a = a.reshape( (a.shape[0]*a.shape[1],8,8)  )
        return a

    def prepare(self):
        return self.flatten()

class SpectralComponent(ImageComponent):
    def __init__(self, X, Y, Q):
        self.Q=Q
        ImageComponent.__init__(self,X,Y,False)
        self.data = [ [ np.zeros(64).astype(int) for _ in range(self.X)] for __ in range(self.Y)]
        #self.check = [ [ np.zeros(64).astype(int) for _ in range(self.X)] for __ in range(self.Y)]

    def set_data(self, data):
        self.data = np.array(data).reshape((self.Y,self.X,64))
        for x in range(self.X):
            for y in range(self.Y):
                #print self.data[y][x]
                assert self.data[y][x].shape[0]==64
        
    def set_spectral(self,x,y,s,val,DC=False):
        if x >=0 and x < self.X and y>=0 and y < self.Y and s>=0 and s<64:
            #assert self.data[y][x][s] == 0
            #self.check[y][x][s] = 1
            self.data[y][x][s] = val
        else:
            if not DC: 
                print("x={} y={} s={} val={}".format(x,y,s,val))
                assert x < self.X and y < self.Y and s < 64
        
    def get_spectral(self,x,y,s,DC=False):
        if x >=0 and x < self.X and y>=0 and y < self.Y and s>=0 and s<64:
            return self.data[y][x][s]
        else:
            if not DC:
                print("ILLEGAL({},{}): x={} y={} s={}".format(self.X,self.Y,x,y,s))
                assert False
            return 1
                        
    def unflatten_idct2(self):        
        for y in range(self.Y):
            for x in range(self.X):
                self.data[y][x] = unflatten_and_IDCT_block(self.data[y][x],self.Q)

    def prepare(self):
        self.unflatten_idct2()
        return self.flatten()
