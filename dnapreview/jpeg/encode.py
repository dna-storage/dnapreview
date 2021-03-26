from PIL import Image
from dnapreview.jpeg.DCT import dQ,DCT_and_flatten,unflatten_and_IDCT
import numpy as np
import bitarray as ba
from bitarray import bitarray
#from dnastorage.codec.huffman_table import *
import random
import unittest
from dnapreview.jpeg.jpeg import JPEG
from dnapreview.jpeg.jpeg_huffman import JPEGHuffmanTable
from dnapreview.jpeg.codec_helper import _get_int_from_bits,nearest,_get_int,byte_packer,byte_unpacker
from dnapreview.jpeg.codec_helper import *

import logging

logger = logging.getLogger('dna.preview.jpeg.encode')
logger.addHandler(logging.NullHandler())


def differential_encode(array):
    prev = 0
    for i,a in enumerate(array):
        tmp = a[0]
        a[0] -= prev
        prev = tmp

def differential_decode(array):
    prev = 0
    for i,a in enumerate(array):
        a[0] += prev
        prev = a[0]

class DifferentialEncodingTest(unittest.TestCase):
    def test_differential_encoding(self):
        R = random.Random()
        m = np.zeros((64,64))
        for i in range(64):
            for j in range(64):
                m[i][j] = R.randint(0,256)
        print(m)
        copy = m.copy()
        differential_encode(m)
        print(m)
        differential_decode(m)
        print(m)
        assert np.sum(m - copy) == 0


def SOF(self, f):
    return

def Skip(self, f):
    return

def DQT(self, f):
    return

def APP(self, f):
    return

def COM(self, f):
    return

MARKER = {
    0xFFC0: ("SOF0", "Baseline DCT", SOF),
    0xFFC1: ("SOF1", "Extended Sequential DCT", SOF),
    0xFFC2: ("SOF2", "Progressive DCT", SOF),
    0xFFC3: ("SOF3", "Spatial lossless", SOF),
    0xFFC4: ("DHT", "Define Huffman table", Skip),
    0xFFC5: ("SOF5", "Differential sequential DCT", SOF),
    0xFFC6: ("SOF6", "Differential progressive DCT", SOF),
    0xFFC7: ("SOF7", "Differential spatial", SOF),
    0xFFC8: ("JPG", "Extension", None),
    0xFFC9: ("SOF9", "Extended sequential DCT (AC)", SOF),
    0xFFCA: ("SOF10", "Progressive DCT (AC)", SOF),
    0xFFCB: ("SOF11", "Spatial lossless DCT (AC)", SOF),
    0xFFCC: ("DAC", "Define arithmetic coding conditioning", Skip),
    0xFFCD: ("SOF13", "Differential sequential DCT (AC)", SOF),
    0xFFCE: ("SOF14", "Differential progressive DCT (AC)", SOF),
    0xFFCF: ("SOF15", "Differential spatial (AC)", SOF),
    0xFFD0: ("RST0", "Restart 0", None),
    0xFFD1: ("RST1", "Restart 1", None),
    0xFFD2: ("RST2", "Restart 2", None),
    0xFFD3: ("RST3", "Restart 3", None),
    0xFFD4: ("RST4", "Restart 4", None),
    0xFFD5: ("RST5", "Restart 5", None),
    0xFFD6: ("RST6", "Restart 6", None),
    0xFFD7: ("RST7", "Restart 7", None),
    0xFFD8: ("SOI", "Start of image", None),
    0xFFD9: ("EOI", "End of image", None),
    0xFFDA: ("SOS", "Start of scan", Skip),
    0xFFDB: ("DQT", "Define quantization table", DQT),
    0xFFDC: ("DNL", "Define number of lines", Skip),
    0xFFDD: ("DRI", "Define restart interval", Skip),
    0xFFDE: ("DHP", "Define hierarchical progression", SOF),
    0xFFDF: ("EXP", "Expand reference component", Skip),
    0xFFE0: ("APP0", "Application segment 0", APP),
    0xFFE1: ("APP1", "Application segment 1", APP),
    0xFFE2: ("APP2", "Application segment 2", APP),
    0xFFE3: ("APP3", "Application segment 3", APP),
    0xFFE4: ("APP4", "Application segment 4", APP),
    0xFFE5: ("APP5", "Application segment 5", APP),
    0xFFE6: ("APP6", "Application segment 6", APP),
    0xFFE7: ("APP7", "Application segment 7", APP),
    0xFFE8: ("APP8", "Application segment 8", APP),
    0xFFE9: ("APP9", "Application segment 9", APP),
    0xFFEA: ("APP10", "Application segment 10", APP),
    0xFFEB: ("APP11", "Application segment 11", APP),
    0xFFEC: ("APP12", "Application segment 12", APP),
    0xFFED: ("APP13", "Application segment 13", APP),
    0xFFEE: ("APP14", "Application segment 14", APP),
    0xFFEF: ("APP15", "Application segment 15", APP),
    0xFFF0: ("JPG0", "Extension 0", None),
    0xFFF1: ("JPG1", "Extension 1", None),
    0xFFF2: ("JPG2", "Extension 2", None),
    0xFFF3: ("JPG3", "Extension 3", None),
    0xFFF4: ("JPG4", "Extension 4", None),
    0xFFF5: ("JPG5", "Extension 5", None),
    0xFFF6: ("JPG6", "Extension 6", None),
    0xFFF7: ("JPG7", "Extension 7", None),
    0xFFF8: ("JPG8", "Extension 8", None),
    0xFFF9: ("JPG9", "Extension 9", None),
    0xFFFA: ("JPG10", "Extension 10", None),
    0xFFFB: ("JPG11", "Extension 11", None),
    0xFFFC: ("JPG12", "Extension 12", None),
    0xFFFD: ("JPG13", "Extension 13", None),
    0xFFFE: ("COM", "Comment", COM)
}


class JPEGCodec:
    codeword = { 'DHT' : 0xFFC4,
                 'SOF2' : 0xFFC2, # Differential Sequential
                 'SOF5' : 0xFFC5, # Differential Sequential
                 'SOF6' : 0xFFC6, # C6 Differential Progressive
                 'SOI' : 0xFFD8,
                 'EOI' : 0xFFD9,
                 'SOS' : 0xFFDA,
                 'DQT' : 0xFFDB,
                 'DNL' : 0xFFDC,
                 'DRI' : 0xFFDD,
                 'APP' : 0xFFE0,
                 'COM' : 0xFFEE }


    def encode_jfif_app0_header(self):
        app0_order = [ 'length', 'id', 'version', 'units', 'Xdensity',
                       'Ydensity', 'Xthumbnail', 'Ythumbnail', 'data' ]
        ah = { 'length': 16,
                  'id':40,
                  'version':16,
                  'units':8,
                  'Xdensity':16,
                  'Ydensity':16,
                  'Xthumbnail':8,
                  'Ythumbnail':8}

        pack = [ [16,0xFFE0],
                 [ah['length'],16],
                 [ah['id'],"JFIF"],
                 [8,0],
                 [ah['version'],0x0102],
                 [ah['units'],0],
                 [ah['Xdensity'],1],
                 [ah['Ydensity'],1],
                 [ah['Xthumbnail'],0],
                 [ah['Ythumbnail'],0]]

        s = byte_packer(pack[0:2])
        s += b"JFIF"
        s += byte_packer(pack[3:])
        return s
        
    def encode_quantization_table(self, Pq, Tq, table):
        table_order = [ 'DQT', 'Lq', 'Pq', 'Tq' ]
        th = { 'DQT':16, 'Lq':16, 'Pq':4, 'Tq':4 }
        size = len(table) + 3
        pack = [ [th['DQT'],JPEGCodec.codeword['DQT']],
                 [th['Lq'],size],
                 [th['Pq'],Pq],
                 [th['Tq'],Tq]]
        s = byte_packer(pack)
        return s+table

    def decode_quantization_table(self, data):
        table_order = [ 'Lq', 'Pq', 'Tq' ]
        th = { 'DQT':16, 'Lq':16, 'Pq':4, 'Tq':4 }
        unpack = [[th['Lq'],0],
                  [th['Pq'],0],
                  [th['Tq'],0]]
        un = byte_unpacker(data,unpack)
        qt ={}
        for to,u in zip(table_order,un):
            qt[to] = u[1]
        qt['table'] = np.array([ord(x) for x in data[3:qt['Lq']]]).reshape((8,8))
        return qt

    def decode_application_string(self, data):
        table_order = [ 'La', ]
        th = { 'APP':16, 'La':16 }
        unpack = [[th['La'],0]]
        un = byte_unpacker(data,unpack)
        appt ={}
        for to,u in zip(table_order,un):
            appt[to] = u[1]
        appt['table'] = np.array([ord(x) for x in data[2:appt['La']]])
        return appt

    
    def encode_huffman_table(self, Tc, Th, table):
        table_order = [ 'DHT', 'Lh', 'Tc', 'Th' ]
        th = { 'DHT':16, 'Lh':16, 'Tc':4, 'Th':4 }
        size = len(table) + 3
        pack = [ [th['DHT'],JPEGCodec.codeword['DHT']],
                 [th['Lh'],size],
                 [th['Tc'],Tc],
                 [th['Th'],Th]]
        s = byte_packer(pack)

        logger.debug("Create Huffman table Tc={} Th={}".format(Tc,Th))

        return s+table

    def decode_huffman_table(self, data):
        table_order = [ 'Lh', 'Tc', 'Th' ]
        th = { 'DHT':16, 'Lh':16, 'Tc':4, 'Th':4 }
        unpack = [ [th['Lh'],0],
                   [th['Tc'],0],
                   [th['Th'],0] ]
        un = byte_unpacker(data,unpack)
        huff ={}
        for to,u in zip(table_order,un):
            huff[to] = u[1]
        huff['table'] = data[3:huff['Lh']]
        return huff
    
    
    def decode_progressive_frame_header(self,str):
        unpack = [ [16, 0], [8, 0], [16, 0], [16, 0], [8, 0] ]
        frame_order = [ 'Lf', 'P', 'Y', 'X', 'Nf' ] # then CSParams

        frame = {}
        un = byte_unpacker(str,unpack)
        for fo,u in zip(frame_order,un):
            frame[fo] = u[1]

        frame['CSParams'] = []
        
        pos = 8
        for i in range(frame['Nf']):
            unpack = [ [8, 0], [4, 0], [4, 0], [8, 0] ]
            u = byte_unpacker(str[pos:pos+3],unpack)
            frame['CSParams'].append( [ x[1] for x in u ] )            
            pos += 3

        return frame
            

    def encode_progressive_frame_header(self,Y,X,CSParams,P=8,Nf=3):
        fh = { 'SOF':16, 'Lf':16, 'P':8, 'Y':16, 'X':16, 'Nf':8 }    
        frame_order = [ 'SOF', 'Lf', 'P', 'Y', 'X', 'Nf' ] # then CSParams
                
        params = b''
        for C,H,V,Tq in CSParams:
            pack = [ [8, C], [4, H], [4, V], [8, Tq] ]
            params += byte_packer(pack)

        Lf = 8 + len(params)
        
        pack = [ [fh['SOF'],JPEGCodec.codeword['SOF2']],\
                 [fh['Lf'],Lf],
                 [fh['P'],P],
                 [fh['Y'],Y],
                 [fh['X'],X],
                 [fh['Nf'],Nf]]

        return byte_packer(pack) + params


    def encode_restart_interval(self,restart_interval):
        self.restart_interval = restart_interval
        pack = [ [16,0xFFDD], [16,4], [16, restart_interval] ]
        logger.debug("create restart_interval={}".format(restart_interval))
        return byte_packer(pack)
        
    def encode_scan_header(self,Ns,CSParams,Ss,Se,Ah,Al):
        sh = { 'SOS':16, 'Ls':16, 'Ns':8, 'Ss':8, 'Se':8, 'Ah':4, 'Al':4}
        scan_order = [ 'SOS', 'Ls', 'Ns', 'CSParams', 'Ss', 'Se', 'Ah', 'Al' ]

        b = b""
        for Cs,Td,Ta in CSParams:
            pack = [ [8, Cs], [4, Td], [4, Ta] ]
            b += byte_packer(pack)

        pack = [ [ sh['Ss'], Ss],
                 [ sh['Se'], Se ],
                 [ sh['Ah'], Ah ],
                 [ sh['Al'], Al ] ]

        b += byte_packer(pack)

        pack = [ [ sh['SOS'], JPEGCodec.codeword['SOS'] ],
                 [ sh['Ls'], len(b)+3 ],
                 [ sh['Ns'], Ns ]]
            
        return byte_packer(pack) + b
    
    def decode_scan_header(self,data):
        sh = { 'SOS':16, 'Ls':16, 'Ns':8, 'Ss':8, 'Se':8, 'Ah':4, 'Al':4}
        scan_order = [ 'Ls', 'Ns', 'CSParams', 'Ss', 'Se', 'Ah', 'Al' ]

        unpack = [ [ sh['Ls'], 0 ],
                 [ sh['Ns'], 0 ]]

        scan = {}
        un = byte_unpacker(data,unpack)
        for so,u in zip(scan_order[0:2],un):
            scan[so] = u[1]

        params = []
        pos = 3
        for i in range(scan['Ns']):
            unpack = [ [8, 0], [4, 0], [4, 0] ]
            un = byte_unpacker(data[pos:],unpack)
            params.append( [ x[1] for x in un ] )
            pos += 2
        scan['CSParams'] = params        
            
        unpack = [ [ sh['Ss'], 0],
                 [ sh['Se'], 0 ],
                 [ sh['Ah'], 0 ],
                 [ sh['Al'], 0 ] ]

        un = byte_unpacker(data[pos:],unpack)
        for so,u in zip(scan_order[3:],un):
            scan[so] = u[1]
                        
        return scan

    def __init__(self):
        return

AC_magnitude_encoding = [ [x,list(range(-2**x+1,-2**(x-1)+1,1))+list(range(2**(x-1),2**x,1))] \
                              for x in range(1,11) ]
DC_magnitude_encoding = [ [0,[0]] ] + [ [x,list(range(-2**x+1,-2**(x-1)+1,1))+list(range(2**(x-1),2**x,1))] \
                              for x in range(1,12) ]


EOB_magnitude_encoding = [ [x-1,list(range(2**(x-1),2**x,1))] for x in range(1,16) ]


def restart_marker_bytes(m):
    logger.debug("create marker for m={}".format(m))
    return bytes([0xFF,m%8+0xD0])

class JPEGProgressiveEncoder(JPEGCodec):
    
    def get_num_pixels(self):
        return jpeg.image.size[0]*jpeg.image.size[1]

    def _get_huffmantable(self, band):
        h = {}
        for b in np.nditer(band):
            h[int(b)] = h.get(int(b),0) + 1
        array = list(h.items())
        assert len(array) > 0
        vals = [a[0] for a in array]
        weights = [a[1] for a in array]
        #w = weights[:]
        #w.sort()
        #print w
        weights = [ float(w)/sum(weights) for w in weights ]
        return JPEGHuffmanTable(vals,weights) #LengthLimitedHuffmanTable(16,2,['0','1'],vals,weights)
    

    def encode_as_byte_string():
        return

                                   
    def select_EOB_magnitude_encoding(self,b):
        global EOB_magnitude_encoding
        b = int(b)
        for i,s in EOB_magnitude_encoding:
            if b in s:
                return i
        assert 0 and "Should not have failed here!"
            
    def select_magnitude_encoding(self,b,DC=False):
        global AC_magnitude_encoding
        global DC_magnitude_encoding
        b = int(b)
        if DC==False: #AC
            for i,s in AC_magnitude_encoding:
                if b in s:
                    return i
            assert 0 and "Should not have failed here!"
        else:
            for i,s in DC_magnitude_encoding:
                if b in s:
                    return i
            assert 0 and "Should not have failed here!"
            

    def decode_low_bits(self,val,nbits):
        #print val,nbits
        v = _get_int_from_bits(val)
        if (v) & (1 << (nbits-1)):
            return v
        else:
            return (-1 ^ (2**nbits-1) ^ v ) + 1 
            
    def encode_low_bits(self,val,nbits):
        val = int(val)
        assert val!=0
        if val > 0:
            o = val & (2**nbits-1)
        else:
            o = (val-1) & (2**nbits-1)
        return bitarray("{:0{w}b}".format(o,w=nbits))

    def prepare_entropy_decode(self,eband,DC=False):
        dband = []
        for RRRRSSSS,val in eband:
            z = RRRRSSSS/16
            m = RRRRSSSS & 0xF
            if m==0 and z==0:
                break
            elif m==0 and z==15:
                dband += [0]*16
            else:
                dband += [0]*z
                if m > 0:
                    dband += [self.decode_low_bits(val,m)]
        return dband

        # assumes one band at a time
    def prepare_entropy_encode_DC(self,band):
        enc = []
        R = 0
        for b in band:
            SSSS = self.select_magnitude_encoding(b,DC)
            #print "SSSS={} RRRR={} = {}".format(SSSS,R*16,RRRRSSSS)
            assert SSSS < 16
            if SSSS == 0:                
                enc.append( [ SSSS, bitarray(0) ] )
            else:
                enc.append( [ SSSS, self.encode_low_bits(b,SSSS) ] )
        return enc                            

    # assumes one band at a time
    def prepare_entropy_encode(self,band,DC=False):
        if DC:
            return self.prepare_entropy_encode_DC(band)

        enc = []
        R = 0
        for b in band:
            if b == 0:
                R += 1
            else:
                while R >= 16:
                    enc.append( [ 15*16, bitarray(0) ] )
                    R -= 16
                assert R <= 15 and R>=0
                SSSS = self.select_magnitude_encoding(b,DC)
                RRRRSSSS = R*16+SSSS
                #print "SSSS={} RRRR={} = {}".format(SSSS,R*16,RRRRSSSS)
                assert RRRRSSSS < 256
                R = 0
                enc.append( [ RRRRSSSS, self.encode_low_bits(b,SSSS) ] )
        if R > 0:
            while R >= 16:
                enc.append( [ 15*16, bitarray(0) ] )
                R -= 16
            if R > 0:
                if R==15:
                    enc.append( [ 14*16, bitarray(0) ] )
                    enc.append( [ 1*16, bitarray(0) ] )
                else:
                    enc.append( [ R*16, bitarray(0) ] )
        enc.append( [0,bitarray(0)] )
        return enc                            

    def decode_EOBRUN(self,bits,m):
        #print "decode_EOBRUn bits={} m={}".format(bits,m)
        if m==0:
            return 1
        else:
            low = _get_int_from_bits(bits)
            low += 2**m
            return low
    
    def encode_EOBRUN(self,EOBRUN):
        SSSS = self.select_EOB_magnitude_encoding(EOBRUN)
        RRRRSSSS = SSSS*16
        if SSSS == 0:
            low = bitarray(0)
        else:
            low = self.encode_low_bits(EOBRUN,SSSS)
        #print "encode: EOBRUN={} {}".format(EOBRUN,[RRRRSSSS,low])
        return [RRRRSSSS,low]


    def prepare_progressive_entropy_DC_encode(self,band,Ss,Se):
        assert Ss==0 and Se==0
        enc = []
        rCount = 0
        marker = 0
        last = 0
        for i,b in enumerate(band):
            # inject markers as desired
            if self.restart_interval > 0:
                if i>0 and rCount == self.restart_interval:
                    logger.debug("Insert marker {} before {}".format(marker,i))
                    enc.append( [ -1, marker ] )
                    marker = (marker+1)%8
                    rCount = 0
                    # reset differential encoding
                    last = 0
                rCount += 1

            logger.debug("DC i={} val={}".format(i,b))
                
            # perform differential encoding
            tmp = b
            b = b - last
            last = tmp
                
            SSSS = self.select_magnitude_encoding(b,True)
            assert SSSS < 16
            if SSSS > 0:
                enc.append( [ SSSS, self.encode_low_bits(b,SSSS) ] )
            else:
                enc.append( [ SSSS, bitarray(0) ] )
        return enc

    
    # only one band at a time is required by JPEG spec
    # UGLY: to handle restart markers, since RRRRSSSS is never allowed to be negative, we
    # simply insert the restart marker directly in this list, with the first value being -1,
    # and second value being the marker number
    def prepare_progressive_entropy_encode(self,band,Ss,Se,DC=False):        
        if DC:
            return self.prepare_progressive_entropy_DC_encode(band,Ss,Se)

        # count between retart intervals, all codes from Ss to Se considered a single
        # MCU, so we only count once every time we reach Se
        rCount = 0
        marker = 0
        
        
        enc = []
        R = 0
        EOBRUN = 0
        K = Ss-1
        for b in band:
            K += 1
            if b==0:
                R = R+1
                if K == Se:
                    EOBRUN += 1
            else:
                if EOBRUN > 0:
                    enc.append( self.encode_EOBRUN(EOBRUN) )
                    EOBRUN = 0
                while R >= 16:
                    enc.append( [ 15*16, bitarray(0) ] )
                    R -= 16
                assert R <= 15 and R>=0
                SSSS = self.select_magnitude_encoding(b,DC)
                RRRRSSSS = R*16+SSSS
                assert RRRRSSSS < 256
                R = 0
                enc.append( [ RRRRSSSS, self.encode_low_bits(b,SSSS) ] )

            # handle of end of band scenario
            if K >= Se:
                rCount += 1
                if self.restart_interval > 0 and rCount == self.restart_interval:
                    if EOBRUN > 0:
                        enc.append( self.encode_EOBRUN(EOBRUN) )
                        EOBRUN = 0
                    rCount = 0
                    enc.append( [ -1, marker ] )
                    marker = (marker+1)%8                                    
                elif EOBRUN == 0x7FFF:
                    enc.append( self.encode_EOBRUN(EOBRUN) )
                    EOBRUN = 0
                K = Ss-1 # reset
                R = 0    # reset
        if EOBRUN > 0:
            enc.append( self.encode_EOBRUN(EOBRUN) ) 
        return enc                            

    def get_bits_to_decode(self, b, l=16):
        if b.length() < l:
            return b.to01()
        else:
            return b[0:l].to01()
        
    
        # assumes one band at a time
    def prepare_progressive_entropy_decode(self,band,ht,Ss,Se,DC=False):
        b = bitarray()
        #print ht.get_tables()[0]
        band = self.removePaddingAfterFF(band)
        b.frombytes(band)
        dband = []
        K = Ss - 1
        while True:
            #print dband
            K = K+1
            #print "K={}".format(K)
            l,RRRRSSSS = ht.decode(self.get_bits_to_decode(b))
            if l==0:
                break
            m = RRRRSSSS & 0xF
            z = RRRRSSSS / 16
            val = b[l:l+m]
            if m==0 and z==0:
                while K <= Se:
                    dband += [0]
                    K += 1
                #print "EOB0 K={}".format(K)
                b = b[l+m:]
            elif m==0 and z==15:
                # ZRL append 16 zeroes
                dband += [0]*16
                K += 16-1
                #print "ZRL K={}".format(K)
                b = b[l+m:]
            elif m==0 and z>=0 and z<=14:
                val = b[l:l+z]
                eobrun = self.decode_EOBRUN(val,z)
                # finish current run
                while K <= Se:
                    dband += [0]
                    K += 1
                #print "EOBn K={} EOBRUN={}".format(K,eobrun)
                dband += [0]*((Se-Ss+1)*(eobrun-1))
                b = b[l+z:]
            else:
                dband += [0]*z
                assert m > 0
                dband += [self.decode_low_bits(val,m)]
                #print "Other K={} z={} val={}".format(K,z,self.decode_low_bits(val,m))
                K += z
                assert K <= Se+1
                b = b[l+m:]
                
            if K >= Se:
                K = Ss-1                                
                #print "Reset K={}".format(K)
        return dband




    
    def padFFWithZero(self, raw):
        raw = raw.replace(b'\xff',b'\xff\x00')
        return raw
        
    def getSectionBoundary(self, res):
        i = 0
        while i < len(res):
            if res[i]==chr(255) and res[i+1]!=chr(0):
                break
            i = i+1
        return i
            
    def removePaddingAfterFF(self, raw):
        raw = raw.replace('\xff\x00','\xff')
        return raw
        
    def entropy_encode_better2(self,p_band,ht):
        #print p_band
        res = b''
        tmp_res = bitarray()
        #print enc
        for k,v in p_band:
            if k==-1:
                logger.debug("Found restart marker {}.".format(v))
                # insert restart marker!
                # markers are always at byte boundaries
                if tmp_res.length()%8 != 0:
                    add = 8 - tmp_res.length()%8
                    tmp_res += bitarray( '1'*add )
                    logger.debug("Pad array {}".format(bitarray( '1'*add ).to01()))

                tmp_res = self.padFFWithZero(tmp_res.tobytes())
                res += tmp_res + restart_marker_bytes(v)
                btmp = bitarray()
                btmp.frombytes(restart_marker_bytes(v))
                logger.debug("Add marker {}".format(btmp.to01()))
                tmp_res = bitarray()
            else:
                tmp_res += bitarray(ht.encode(k))+v

        if tmp_res.length()%8 != 0:
            add = 8 - tmp_res.length()%8
            tmp_res += bitarray( '1'*add )
        assert tmp_res.length()%8==0
        res = res + self.padFFWithZero(tmp_res.tobytes())
        #res = res.tobytes()
        if False:
            newht = JPEGHuffmanTable.decode_table(ht.encode_table())
            #print ht.get_raw_table()
            #print newht.get_raw_table()
            #assert newht.get_raw_table()==ht.get_raw_table()
            data = self.entropy_decode_better(res,newht)
            #print data
            p = self.prepare_entropy_encode(data,True)
            for a,b in zip(p_band,p):
                #print "Check: {}={}".format(a,b)
                assert a==b
                #assert True
            
        return res

    def entropy_decode_better(self,band,ht):
        b = bitarray()
        #print ht.get_tables()[0]
        band = self.removePaddingAfterFF(band)
        b.frombytes(band)
        dband = []
        while True:
            l,RRRRSSSS = ht.decode(b[0:16].to01())
            m = RRRRSSSS & 0xF
            z = RRRRSSSS / 16
            val = b[l:l+m]
            if m==0 and z==0:
                b = b[l+m:]
                break
            elif m==0 and z==15:
                dband += [0]*16
            else:
                dband += [0]*z
                if m > 0:
                    dband += [self.decode_low_bits(val,m)]
            b = b[l+m:]
        return dband
            

    def entropy_encode(self,band,ht):
        # assume that band is a one-dimensional array that has already
        # been appropriately flattened
        res = bitarray()
        enc,dec = ht.get_tables()                
        tb = np.trim_zeros(band,'b')
        z = 0
        for t in tb:
            if t==0 and z < 15:
                z+=1
                # emit nothing
            elif t==0 and z==15:
                z = 1
                res += "11110000"
                # emit special byte that means 15 zeros
            else:
                # emit byte + huffman code
                s = enc[t]
                assert len(s) <= 16
                tmp = "{0:{fill}4b}".format(z, fill='0')
                tmp += "{0:{fill}4b}".format(len(s), fill='0')
                res += tmp + s
                z = 0

        res += "00000000"

        if True:
            left,d = self.entropy_decode(res.tobytes(),ht,len(band))
            d = np.array(d)
            for i,(x,y) in enumerate(zip( [int(b) for b in band], [int(dd) for dd in d] )):
                if x!=y:
                    print("{} x={} y={} {}".format(i,x,y,d[i-16:i+1]))
                    break                    
            assert np.sum(d-band)==0
        
        return res

    def entropy_decode(self,band,ht,samples):
        # assume that band is a one-dimensional array that has already
        # been appropriately flattened
        b = bitarray()
        b.frombytes(band)
        enc,dec = ht.get_tables()                
        arr = []
        pos = 0
        while True:
            num_zeros = _get_int_from_bits(b[pos:pos+4])
            arr += [0] * num_zeros 
            length = _get_int_from_bits(b[pos+4:pos+8])
            #if num_zeros == 15 and length == 0:                
            if num_zeros == 0 and length == 0:
                # found of end of block
                pos += 8
                break
            elif length > 0:
                key = b[pos+8:pos+8+length].to01()
                #print dec
                arr.append( dec[key] )
            pos += 8 + length
        #print "1. len(arr)={} sample={}".format(len(arr),samples)
        if len(arr) < samples:
            arr += [0] * (samples-len(arr))
        #print "2. len(arr)={} sample={}".format(len(arr),samples)

        band = b[pos:].tobytes()
        return band,arr
        
        
    def __init__(self, jpeg, CSParams={1:{'id':1,'Hi':2,'Vi':2,'q':0},\
                                       2:{'id':2,'Hi':1,'Vi':1,'q':0},\
                                       3:{'id':3,'Hi':1,'Vi':1,'q':0}}):
        if jpeg != None:
            self.jpeg = jpeg
            self.CSParams = CSParams
            self.X = self.jpeg.image.size[0]
            self.Y = self.jpeg.image.size[1]
            self.maxH = max( [ CSParams[i]['Hi'] for i in list(CSParams.keys()) ] )
            self.maxV = max( [ CSParams[i]['Vi'] for i in list(CSParams.keys()) ] )
            self.hts_dc = [ None, None, None, None ]
            self.hts_ac = [ None, None, None, None ]
            self.qts = [ None, None, None, None ]
        else:
            self.jpeg = None
        self._have_components = False
        self.comps = [None,None,None,None]
        self.restart_interval = 0

    def get_ds(self, id):
        assert id in self.CSParams
        return ( self.maxH // self.CSParams[id]['Hi'], self.maxV // self.CSParams[id]['Vi'] )

    def get_size(self, id):
        assert id in self.CSParams
        return ( nearest(self.X * self.CSParams[id]['Hi']//self.maxH,8), nearest(self.Y * self.CSParams[id]['Vi']//self.maxV,8) )

    def init_components(self,Q=dQ,force=False):
        if force or self._have_components == False:
            for i in list(self.CSParams.keys()):
                size = self.get_size(i)
                c = SpectralComponent(size[0]//8,size[1]//8,Q)        
                self.comps[i] = c
                if i==1:
                    t = DCT_and_flatten(self.jpeg.get_Y_blocks(self.get_ds(i)),Q)
                    c.set_data(t)
                elif i==2:
                    c.set_data(DCT_and_flatten(self.jpeg.get_Cb_blocks(self.get_ds(i)),Q))
                elif i==3:
                    c.set_data(DCT_and_flatten(self.jpeg.get_Cr_blocks(self.get_ds(i)),Q))            
            self._have_components = True
            #print self.comps
            
    def encode(self):
        return self.default_encode()

    def encode_jfif_header(self,Q=None):
        pack = [[ 16, JPEGCodec.codeword['SOI'] ]]
        b = byte_packer(pack)
        # todo: add thumbnail
        b += self.encode_jfif_app0_header()
        return b

    def encode_end(self):
        pack = [[ 16, JPEGCodec.codeword['EOI'] ]]
        return byte_packer(pack)
    
    def encode_frame_header(self):
        params = [ [d['id'],d['Hi'],d['Vi'],d['q']] for k,d in list(self.CSParams.items()) ]
        params.sort(key=lambda x: x[0])        
        b = self.encode_progressive_frame_header(self.jpeg.image.size[1],self.jpeg.image.size[0],params)
        return b

    def encode_q(self,id,Q=None):        
        if Q==None:
            Q = np.ones( (8,8) )
        b = self.encode_quantization_table(0,0, bytes([ int(x) for x in list(Q.flatten()) ]) )
        self.qts[id] = Q
        return b
        
    def encode_dc_ht(self,id,ht_id,Al=0):
        #print self.comps
        Ss=0
        Se=0
        Ah=0
        Ns=1
        b = []
        size = self.get_size(id)
        arr = []
        last = 0
        for y in range(size[1]//8):
            for x in range(size[0]//8):
                val = (int(self.comps[id].get_spectral(x,y,0)) >> Al) 
                #tmp = val
                #val = val - last
                #last = tmp
                arr.append( val )

        #print (dump)
        prep_band = self.prepare_progressive_entropy_encode(arr,0,0,True)
        ht_vals = [ x[0] for x in prep_band ]
        #print (ht_vals)
        while -1 in ht_vals: # restart markers insert a -1, so remove them!
            ht_vals.remove(-1)
        ht = self._get_huffmantable([ht_vals])
        self.hts_dc[ht_id] = ht
        return self.encode_huffman_table(0,ht_id,ht.encode_table())
    
    def encode_dc_scan(self,id,ht_id,q_id,Al=0):
        Ss=0
        Se=0
        Ah=0
        Ns=1
        b = []
        size = self.get_size(id)
        arr = []
        last = 0
        for y in range(size[1]//8):
            for x in range(size[0]//8):
                val = (int(self.comps[id].get_spectral(x,y,0)) >> Al)
                # differential encoding moved to prepare_progressive_entropy_DC_encode,
                # because it must be done at same time as restart markers
                #tmp = val
                #val = val - last
                #last = tmp
                arr.append( val )

        #print arr
        bttemp = bitarray()                
        prep_band = self.prepare_progressive_entropy_encode(arr,0,0,True)
        bttemp.frombytes(self.encode_scan_header(Ns,[(id,ht_id,0)],Ss,Se,Ah,Al))
        bttemp.frombytes(self.entropy_encode_better2(prep_band,self.hts_dc[ht_id]))
        assert bttemp.length()%8==0
        return bttemp.tobytes()

        
    def encode_ac_scan(self,id,ht_id,q_id,Ss,Se,Al):
        Ns = 1
        Ah = 0
        b = b""
        size = self.get_size(id)
        arr = []
        for y in range(size[1]//8):
            for x in range(size[0]//8):
                for s in range(Ss,Se+1):
                    val = (int(self.comps[id].get_spectral(x,y,s)) >> Al)
                    arr.append( val )
        
        prep_band = self.prepare_progressive_entropy_encode(arr,Ss,Se,False)
        ht_vals = [ x[0] for x in prep_band ]
        while -1 in ht_vals:
            ht_vals.remove(-1)

        ht = self._get_huffmantable([ht_vals])
        b += self.encode_huffman_table(1,ht_id,ht.encode_table())
        b += self.encode_scan_header(Ns,[(id,0,ht_id)],Ss,Se,Ah,Al)
        b += self.entropy_encode_better2(prep_band,ht)        
        return b
        
    
    def default_encode(self):

        scan_buffer = []
        
        myQ = np.ones( (8,8) )

        Y_ds = (1,1)
        Cr_ds = (1,1)
        Cb_ds = (1,1)
        
        Y = DCT_and_flatten(jpeg.get_Y_blocks(),myQ)
        Cr = DCT_and_flatten(jpeg.get_Cr_blocks(Cr_ds),myQ)
        Cb = DCT_and_flatten(jpeg.get_Cb_blocks(Cb_ds),myQ)
        #print Y.shape
        #print Cr.shape
        #print Cb.shape

        pack = [[ 16, JPEGCodec.codeword['SOI'] ]]
        b = byte_packer(pack)
        b += self.encode_jfif_app0_header()

        CSParams = [ (1,1,1,0), (2,Cb_ds[0],Cb_ds[1],0), (3,Cr_ds[0],Cr_ds[1],0) ]

        b += self.encode_quantization_table(0,0, bytes([ int(x) for x in list(myQ.flatten()) ]) )

        b += self.encode_progressive_frame_header(jpeg.image.size[1],jpeg.image.size[0],CSParams)
        
        differential_encode(Y)
        differential_encode(Cr)
        differential_encode(Cb)

        YT = np.transpose(Y)
        CrT = np.transpose(Cr)
        CbT = np.transpose(Cb)

        size = 0

        bt = bitarray()
        bt.frombytes(b)

        scan_buffer.append( bt.tobytes() )

        for j,T in enumerate([YT,CbT,CrT]):
            for i,yt in enumerate(T):
                bttemp = bitarray()                
                prep_band = self.prepare_progressive_entropy_encode(yt,0,0,i==0)
                ht_vals = [ max(0,x[0]) for x in prep_band ]
                Yht = self._get_huffmantable([ht_vals])
                bttemp.frombytes(self.encode_huffman_table(int(i>=1),0,Yht.encode_table()))
                bttemp.frombytes(self.encode_scan_header(1,[(j+1,0,0)],i,i,0,0))
                bttemp.frombytes(self.entropy_encode_better2(prep_band,Yht))
                scan_buffer.append(bttemp.tobytes())
                bt += bttemp
                assert bt.length()%8==0
                                
        pack = [[ 16, JPEGCodec.codeword['EOI'] ]]
        bt.frombytes(byte_packer(pack))

        #total = sum([len(x) for x in scan_buffer])
        #for s in scan_buffer:
        #    print len(s),len(s)/float(total)*100
        
        return bt.tobytes()
        
        
    def decode(self, allbytes):
        unpack = [[16, 0]]
        byte_unpacker(allbytes,unpack)

        self.hts = [None,None,None,None]
        self.Qs  = [None,None,None,None]

        YT = []
        CrT = []
        CbT = []


        comps = [ YT, CbT, CrT ]
        
        if unpack[0][1] != JPEGCodec.codeword['SOI']:        
            assert 0 and "Failed to match SOI 0x{:x}".format(unpack[0][1])
            sys.exit(-1)

        allbytes = allbytes[2:]
        frame_header = {}

        sos_cnt=0
        
        while True:
            unpack = [[16,0]]
            byte_unpacker(allbytes,unpack)
            kind = unpack[0][1]
            if kind == JPEGCodec.codeword['EOI']:
                break # end of image, no more bytes should be read

            print("kind=0x{:x}".format(kind))

            allbytes = allbytes[2:]
            unpack = [[16,0]]
            byte_unpacker(allbytes,unpack)
            L = unpack[0][1]
        
            if len(allbytes) < L:
                print("File is not long enough, need {}B only {}B remain".format(unpack[1][1],len(allbytes)))
                assert 0 
                                
            if JPEGCodec.codeword['DHT'] == kind:
                print("Found DHT")
                hth = self.decode_huffman_table(allbytes)
                #print hth
                self.hts[ hth['Th'] ] = JPEGHuffmanTable.decode_table(hth['table'])
                allbytes = allbytes[L:] # pop of last section
                pass
            elif JPEGCodec.codeword['SOF2'] == kind:
                print("Found SOF2")
                frame_header = self.decode_progressive_frame_header(allbytes)
                allbytes = allbytes[L:] # pop of last section
                print(frame_header)
                pass
            elif JPEGCodec.codeword['SOS'] == kind:
                print("Found SOS {}".format(sos_cnt))
                sos_cnt+=1
                scan_header = self.decode_scan_header(allbytes)
                #print scan_header
                #print scan_header
                allbytes = allbytes[L:] # pop of last section
                #assert scan_header['Ns']==1
                param_id = scan_header[ 'CSParams' ][0][0]-1
                X = nearest(frame_header['X'],8*frame_header['CSParams'][param_id][1])
                Y = nearest(frame_header['Y'],8*frame_header['CSParams'][param_id][2])
                samples = X*Y / ((8*frame_header['CSParams'][param_id][1])*(8*frame_header['CSParams'][param_id][2])) 
                #samples = samples / frame_header['CSParams'][param_id][1] / frame_header['CSParams'][param_id][2]
                #print "Samples={}".format(samples)

                boundary = self.getSectionBoundary(allbytes)
                
                arr = self.prepare_progressive_entropy_decode(allbytes[:boundary], self.hts[ 0 ], scan_header['Ss'], scan_header['Se'])
                allbytes = allbytes[boundary:]
                #print arr
                #print "arr = {}".format(arr)
                #print "allbytes={} arr={}".format(len(allbytes),len(arr))
                #print "arr len = {}".format(len(arr))
                comps[param_id].append(arr)
                pass
            elif JPEGCodec.codeword['DQT'] == kind:
                #print "Found DQT"
                qh = self.decode_quantization_table(allbytes)
                self.Qs[ qh['Tq'] ] = qh['table']
                allbytes = allbytes[L:] # pop of last section
                pass
            elif JPEGCodec.codeword['COM'] == kind:
                #print "Found COM"
                allbytes = allbytes[L:] # pop of last section                            
                pass
            elif JPEGCodec.codeword['APP'] <= kind and JPEGCodec.codeword['APP']+0xF >= kind:
                apph = self.decode_application_string(allbytes)
                print(apph)
                print("".join([chr(x) for x in apph['table']]))
                allbytes = allbytes[L:]
            else:
                print("Do not know how to handle section 0x{:x}".format(kind))
                assert False 


        for i in range(len(comps)):
            comps[i] = np.array(comps[i])

        for k,i in enumerate(comps[0]):
            print(k,len(i))
            
        Y = np.transpose(comps[0])
        Cb = np.transpose(comps[1])
        Cr = np.transpose(comps[2])

        differential_decode(Y)
        differential_decode(Cr)
        differential_decode(Cb)

        print(Y.shape)
        print(Cb.shape)
        print(Cr.shape)
        
        size = ( frame_header['X'], frame_header['Y'] )

        if self.jpeg == None:
            jpeg = JPEG(None,1,size)
            self.jpeg = jpeg

        Y_ds = [frame_header['CSParams'][0][1],frame_header['CSParams'][0][2]]
        Cb_ds = [frame_header['CSParams'][1][1],frame_header['CSParams'][1][2]]
        Cr_ds = [frame_header['CSParams'][2][1],frame_header['CSParams'][2][2]]
            
        self.jpeg.set_Y_blocks(unflatten_and_IDCT(Y),Y_ds)
        self.jpeg.set_Cb_blocks(unflatten_and_IDCT(Cb),Cb_ds)
        self.jpeg.set_Cr_blocks(unflatten_and_IDCT(Cr),Cr_ds)
        self.jpeg.merge()
        #self.jpeg = jpeg

        self.jpeg.show()
        return

    def sanity_check(self):
        myQ = dQ #np.ones( (8,8) )
        Y_ds = (1,1)
        Cr_ds = (2,2)
        Cb_ds = (2,2)        
        Y = DCT_and_flatten(jpeg.get_Y_blocks(),myQ)
        Cr = DCT_and_flatten(jpeg.get_Cr_blocks(Cr_ds),myQ)
        Cb = DCT_and_flatten(jpeg.get_Cb_blocks(Cb_ds),myQ)

        differential_encode(Y)
        differential_encode(Cr)
        differential_encode(Cb)
        differential_decode(Y)
        differential_decode(Cr)
        differential_decode(Cb)
        
        self.jpeg.set_Y_blocks(unflatten_and_IDCT(Y))
        self.jpeg.set_Cb_blocks(unflatten_and_IDCT(Cb),Cb_ds)
        self.jpeg.set_Cr_blocks(unflatten_and_IDCT(Cr),Cr_ds)
        #self.jpeg.set_Y_blocks(jpeg.get_Y_blocks(Y_ds),Y_ds)
        #self.jpeg.set_Cb_blocks(jpeg.get_Cb_blocks(Cb_ds),Cb_ds)
        #self.jpeg.set_Cr_blocks(jpeg.get_Cr_blocks(Cr_ds),Cr_ds)
        self.jpeg.merge()
        #self.jpeg = jpeg
        self.jpeg.show()


    def get_progressive_scans(self,spectral_size,restart_interval=64,comments=False):
        b = []
        comment = []
        
        # everything in init is required for a complete image
        init = b""
        comm = ""
        init += self.encode_jfif_header()
        comm += "jfif"

        init += self.encode_q(0)
        comm += ",Q0"

        self.init_components(self.qts[0])
        init += self.encode_restart_interval(restart_interval)
        init += self.encode_dc_ht(1,0)
        comm += "DRI({}),DC-HT1".format(restart_interval)

        init += self.encode_frame_header()
        comm += ",SOF"
                
        init += self.encode_dc_scan(1,0,0,0)
        comm += ",Y[0]"
        
        comment = [ comm ]
        
        b = [init]
        # up to here is a fuzzy grayscale image
        
        Y_spectrum = self.encode_ac_scan(1,1,0,1,min(63,spectral_size),0)
        comment += [",Y[{}:{}]".format(1,min(63,spectral_size))]

        b += [Y_spectrum]
        
        # up to here is a somewhat clear grayscale image
        
        b += [self.encode_dc_ht(2,1)+self.encode_dc_scan(2,1,0,0)+
              self.encode_dc_ht(3,2)+self.encode_dc_scan(3,2,0,0)]

        comment += [ ",DC-HT2,Cb[0],DC-HT3,Cr[0]" ]
        
        # up to here is a color image with some color distortion

        for j in range(spectral_size+1,63,spectral_size):    
            b += [self.encode_ac_scan(1,1,0,j,min(63,j+spectral_size-1),0)]
            comment += [ ",Y[{}:{}]".format(j,min(63,j+spectral_size-1)) ]
            
        for j in range(1,63,spectral_size):    
            b += [self.encode_ac_scan(2,2,0,j,min(63,j+spectral_size-1),0)]
            comment += [ ",C[{}:{}]".format(j,min(63,j+spectral_size-1)) ]
            tmp = [self.encode_ac_scan(3,3,0,j,min(63,j+spectral_size-1),0)]
            b += tmp
            comment += [ ",Cr[{}:{}]".format(j,min(63,j+spectral_size-1)) ]

        b += [self.encode_end()]
        comment += [ ',EOI' ]
        # up to here is a complete image
        if comments:
            assert len(b)==len(comment)
            return b,comment
        else:
            return b
        
    # def sequential_size(self):
    #     Yht = self._get_huffmantable(self.Y)
    #     Crht = self._get_huffmantable(self.Cr)
    #     Cbht = self._get_huffmantable(self.Cb)        
    #     size = 0
    #     byt = self.encode_band(self.Y,Yht)
    #     size += len( byt )
    #     size += len( self.encode_band(self.Cr,Crht) )
    #     size += len( self.encode_band(self.Cb,Cbht) )

    #     size += len ( Yht.encode() )
    #     size += len ( Crht.encode() )
    #     size += len ( Cbht.encode() )

    #     pixels = self.Y.shape[0]*self.Y.shape[1]
    #     pixels += self.Cr.shape[0]*self.Cr.shape[1]
    #     pixels += self.Cb.shape[0]*self.Cb.shape[1]

    #     print "Sequential Bits/Pixel = {}".format(float(size)*8.0/self.get_num_pixels())
    #     print "Pixels = {} {}".format(pixels,self.get_num_pixels())

    #     return size

    # def progressive_size(self):
    #     YT = np.transpose(self.Y)
    #     CrT = np.transpose(self.Cr)
    #     CbT = np.transpose(self.Cb)

    #     size = 0
    #     for yt in YT:
    #         yt = np.trim_zeros(yt,'b')
    #         if yt.shape[0]==0:
    #             continue

    #         Yht = self._get_huffmantable([yt])
    #         bits = self.encode_band(np.array([yt]),Yht)            
    #         size+=len(Yht.encode()) + len(bits)

    #     for yt in CrT:
    #         yt = np.trim_zeros(yt,'b')
    #         if yt.shape[0]==0:
    #             continue

    #         Yht = self._get_huffmantable([yt])
    #         bits = self.encode_band(np.array([yt]),Yht)            
    #         size+=len(Yht.encode()) + len(bits)

    #     for yt in CbT:
    #         yt = np.trim_zeros(yt,'b')
    #         if yt.shape[0]==0:
    #             continue

    #         Yht = self._get_huffmantable([yt])
    #         bits = self.encode_band(np.array([yt]),Yht)            
    #         size+=len(Yht.encode()) + len(bits)

    #     pixels = self.Y.shape[0]*self.Y.shape[1]
    #     pixels += self.Cr.shape[0]*self.Cr.shape[1]
    #     pixels += self.Cb.shape[0]*self.Cb.shape[1]

    #     print "Progressive Bits/Pixel = {}".format(float(size)*8.0/self.get_num_pixels())
    #     print "Pixels = {} {}".format(pixels,self.get_num_pixels())

    #     return size


class BytePackerTest(unittest.TestCase):
    def test_byte_packer(self):
        print("Try out bytepacker: ")        
        print([ ord(x) for x in byte_packer( [ [16, 0xFFC6], [16, 19], [8, 8], [16, 512], [16, 512], [8, 3],\
                         [ 8, 0 ], [4, 1], [4, 1], [8,0],\
                         [ 8, 1 ], [4, 4], [4, 4], [8,0],\
                         [ 8, 2 ], [4, 4], [4, 4], [8,0]] )])
        pl = [ [16, 0xFFC6], [16, 19], [8, 8], [16, 512], [16, 512], [8, 3],\
           [ 8, 0 ], [4, 1], [4, 1], [8,0],\
           [ 8, 1 ], [4, 4], [4, 4], [8,0],\
           [ 8, 2 ], [4, 4], [4, 4], [8,0]]
        pl2 = byte_unpacker(byte_packer(pl),pl)
        print("pl={}".format(pl))
        print("pl2={}".format(pl2))
        assert pl==pl2
    
    
class EntropyEncodingTest(unittest.TestCase):
    def test_entropy_encoding(self):
        jpeg = JPEG("/Users/jtuck/Downloads/JPEG_example_JPG_RIP_100.jpg", 1 )
        codec = JPEGProgressiveEncoder(jpeg)
        array = np.array([ np.random.randint(256) for x in range(64)])
        ht = codec._get_huffmantable(array)
        e,d = ht.get_tables()
        print(d)        
        b = codec.entropy_encode(array,ht)
        print(len(b))
        x,arr=codec.entropy_decode(b.tobytes(),ht,64)
        print(array)
        print(arr)
        print("x={}".format(x))        
        assert sum(array - arr)==0

    def test_encode_low_bits(self):
        codec = JPEGProgressiveEncoder(None)
        r = 2
        for i in range(-2**r+1,2**r,1):
            if i!=0:
                c = codec.encode_low_bits(i,codec.select_magnitude_encoding(i,False))
                d = codec.decode_low_bits(c,codec.select_magnitude_encoding(i,False))
                print(i,c,d)
                assert i==d

    def test_prepare_entropy_encode(self):
        codec = JPEGProgressiveEncoder(None)
        arr = np.array([ np.random.randint(16) for x in range(20)])
        #print codec.prepare_entropy_encode(arr,True)
        a =  codec.prepare_entropy_encode( arr ,True)
        print(a)
        res = codec.prepare_entropy_decode(a, True)
        print(arr)
        print(res)
        assert np.sum(arr-np.array(res))==0

    def test_progressive_entropy_encode(self):
        codec = JPEGProgressiveEncoder(None)
        Ss = 1
        Se_vals = [1]+[ np.random.randint(2,63) for _ in range(5) ]
        for Se in Se_vals:
            l = (Se-Ss+1)*10
            for i in range(1,l-1,2):
                arr = np.zeros( l )
                arr[0:l:i] = [ np.random.randint(255) for x in range(0,l,i) ]
                print("Initial array = \n{}".format(arr))            
                #print codec.prepare_entropy_encode(arr,True)
                prep_band =  codec.prepare_progressive_entropy_encode( arr ,Ss,Se,False)
                #print prep_band
                ht_vals = [ max(0,x[0]) for x in prep_band  ]
                Yht = codec._get_huffmantable([ht_vals])
                #print Yht.get_tables()[0]        
                enc = codec.entropy_encode_better2(prep_band,Yht)
                #print len(enc)
                #print [ord(x) for x in enc]
                dec_band = codec.prepare_progressive_entropy_decode(enc,Yht,Ss,Se,False)        
                print(np.array(dec_band))
                assert np.sum(np.array(dec_band)-arr)==0
                print("SUCCESS i={}".format(i))

        
if __name__ == "__main__":
    import sys
    #from dnapreview.jpeg.encode import *
    from dnapreview.logger import logger
    for ri in [0,16,128]:
        for rm in [60]:
            for z in [10]:
                jpeg = JPEG( sys.argv[1], 5 )
                codec = JPEGProgressiveEncoder(jpeg)

                b  = codec.encode_jfif_header()
                b += codec.encode_q(0)
                codec.init_components(codec.qts[0])
                b += codec.encode_restart_interval(ri)
                b += codec.encode_dc_ht(1,0)
                b += codec.encode_dc_ht(2,1)
                b += codec.encode_dc_ht(3,2)
                b += codec.encode_frame_header()
                b += codec.encode_dc_scan(1,0,0,0)
                #b += ''.join([chr(0xFF)*1000])
                b += codec.encode_dc_scan(2,1,0,0)
                b += codec.encode_dc_scan(3,2,0,0)
                #b += ''.join([chr(0xFF)*1000])

                b += codec.encode_restart_interval(ri)
                for j in range(1,rm,z):    
                    b += codec.encode_ac_scan(1,1,0,j,min(63,j+z-1),0)
                    b += codec.encode_ac_scan(2,2,0,j,min(63,j+z-1),0)
                    b += codec.encode_ac_scan(3,3,0,j,min(63,j+z-1),0)

                b += codec.encode_end()

                name = "wuflab_z{}_rm{}_ri{}.jpg".format(z,rm,ri)
                f = open(name,"wb")
                f.write(b)
                f.close()

    #decodec = JPEGProgressiveCodec(None)
    #decodec.decode(res)

    #f = open(sys.argv[1], "rb")
    #res = f.read()
    #decodec.decode(res)
    
    
    sys.exit(0)

    

    jpeg = JPEG( sys.argv[1], 1 )

    opt = JPEGEncoder(jpeg)
    print(opt.sequential_size())
    print(opt.progressive_size())

    sys.exit(0)


    print(jpeg.Y.getbands())
    #print enc.Cb
    #print enc.Cr


    from dnastorage.codec.huffman_table import *

    def split(band,th=2):
        band = DCT_and_flatten(band)
        T = np.transpose(band)
        
        remove = []
        for j,t in enumerate(T):
            h = { i : 0 for i in t }
            for i in t:
                h[i] += 1
        
            ht = LengthLimitedHuffmanTable(16,2,['0', '1'], list(h.keys()), list(h.values()))
            #print "{}: {} {:.0f}".format(j,ht.average_length(),Y.shape[0]*(ht.average_length()+1)/8.0)
        
            if ht.average_length() < th:
                remove.append(j)

            for r in remove:
                T[r] = np.zeros( T[r].shape[0] )

        band = np.transpose(T)
        band = unflatten_and_IDCT(band)
        return band

    jpeg.set_Y_blocks( split(jpeg.get_Y_blocks(),3) )
    jpeg.set_Cb_blocks( split(jpeg.get_Cb_blocks(),3) )
    jpeg.set_Cr_blocks( split(jpeg.get_Cr_blocks(),3) )

    jpeg.merge()
    jpeg.show()
