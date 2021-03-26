from PIL import Image
from dnapreview.jpeg.DCT import dQ,DCT_and_flatten,unflatten_and_IDCT_block,unflatten_by_zig_zag,idct2,dct2
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

from dnastorage.codec.huffman_table import ErrorWhileDecodingTable

import logging
logger = logging.getLogger('dna.decoder')
#logger.addHandler(logging.NullHandler())
    
def SOF(self, b):
    unpack = [[16,0]]
    byte_unpacker(b,unpack)
    return b[unpack[0][1]:]

def frame_header(self, data):
    unpack = [ [16, 0], [8, 0], [16, 0], [16, 0], [8, 0] ]
    frame_order = [ 'Lf', 'P', 'Y', 'X', 'Nf' ] # then CSParams
    
    frame = {}
    un = byte_unpacker(data,unpack)
    for fo,u in zip(frame_order,un):
        frame[fo] = u[1]
        
    frame['CSParams'] = []
    
    pos = 8
    for i in range(frame['Nf']):
        unpack = [ [8, 0], [4, 0], [4, 0], [8, 0] ]
        u = byte_unpacker(data[pos:pos+3],unpack)
        frame['CSParams'].append( [ x[1] for x in u ] )            
        pos += 3

    self.maxH = max( [x[1] for x in frame['CSParams']] )
    self.maxV = max( [x[2] for x in frame['CSParams']] )


    
    self.Y = frame['Y']
    self.X = frame['X']

    logger.debug("Image has size ({},{}).".format(self.X,self.Y))

    self.Nf = frame['Nf']
    logger.debug("Image has {} components.".format(self.Nf))
        
    self.CSParams = { i:{'Hi':dsx, 'Vi':dsy, 'q':q} for i,dsx,dsy,q in frame['CSParams'] }

    adjY = nearest(self.Y,8)//8
    adjX = nearest(self.X,8)//8
    
    for id,params in list(self.CSParams.items()):
        self.CSParams[id]['adjX'] = adjX*params['Hi']/self.maxH
        self.CSParams[id]['adjY'] = adjY*params['Vi']/self.maxV
        self.CSParams[id]['X'] = nearest(self.X*params['Hi']//self.maxH,8)//8
        self.CSParams[id]['Y'] = nearest(self.Y*params['Vi']//self.maxV,8)//8
        self.CSParams[id]['rX'] = self.maxH//params['Hi']
        self.CSParams[id]['rY'] = self.maxV//params['Vi']
        #self.comps[id] = ImageComponent(self.CSParams[id]['X'],self.CSParams[id]['Y'])
        # self.comps[id] = [[ None for _ in range(self.CSParams[id]['adjX']) ] for _ in range(self.CSParams[id]['adjY'])] 

    logger.debug("{}".format(self.CSParams))
        
    #print self.CSParams
    #print self.X,self.Y
    return data[pos:]

def SOF1(self, data):
    # Extendend Sequential DCT mainly just has more tables
    # and SOF0 is already designed to cope with that.
    for id,params in list(self.CSParams.items()):
        self.comps[id] = ImageComponent(self.CSParams[id]['X'],self.CSParams[id]['Y'])
    return SOF0(self,data)

def SOF0(self, data):
    data = frame_header(self, data)
    self.scan = SOS_sequential_scan
    for id,params in list(self.CSParams.items()):
        self.comps[id] = ImageComponent(self.CSParams[id]['X'],self.CSParams[id]['Y'])
    return data

def SOF2(self, data):
    data = frame_header(self, data)
    #print data
    self.scan = SOS_progressive_scan
    self.spectral_selection = [ [ False for _ in range(64) ] for __ in range(4) ]
    for id,params in list(self.CSParams.items()):
        self.comps[id] = SpectralComponent(self.CSParams[id]['X'],self.CSParams[id]['Y'],self.qts[self.CSParams[id]['q']])
    return data

def scan_header(self,data):
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

    #print scan['CSParams']
            
    unpack = [ [ sh['Ss'], 0],
               [ sh['Se'], 0 ],
               [ sh['Ah'], 0 ],
            [ sh['Al'], 0 ] ]

    un = byte_unpacker(data[pos:],unpack)
    for so,u in zip(scan_order[3:],un):
        scan[so] = u[1]

    msg = [ "{}:{}".format(k,v) for k,v in list(scan.items()) ]
        
    logger.debug(",".join(msg))
        
    #print scan
    return scan

        
def get_bits_to_decode(b, l=16):
    if b.length() < l:
        return b.to01()
    else:
        return b[0:l].to01()

def decode_low_bits(val,nbits):
    #print val,nbits
    v = _get_int_from_bits(val)
    if (v) & (1 << (nbits-1)):
        return v
    else:
        return (-1 ^ (2**nbits-1)) ^ v + 1  

def entropy_decode_block(dcht,acht,b):
    block = []
    ht = dcht
    dc = True
    i = 0
    while len(block)<64:
        l,RRRRSSSS = ht.decode(get_bits_to_decode(b[i:i+16]))
        if l==0:
            return l,None
        z = RRRRSSSS/16
        m = RRRRSSSS & 0xF
        val = b[i+l:i+l+m]
        #b = b[l+m:]
        i += (l+m)
        if dc:
            #print "****",l,z,m,val
            assert z==0
            assert m!=15
            if m==0:
                block += [0]
            else:
                pp = decode_low_bits(val,m)
                #print "DC: val={} m={} res={}".format(val,m,pp)
                block += [pp]
            ht = acht
            dc = False
        else:
            if z==0 and m==0:
                block += [0]*(64-len(block))
                break
            elif z==15 and m==0:
                block += [0]*16
            else:
                block += [0]*z
                if m > 0:
                    #print "AC: val={} m={} res={}".format(val,m,decode_low_bits(val,m))
                    block += [decode_low_bits(val,m)]

    #print len(block),block    
    assert len(block)==64
    return i,block

def removePaddingAfterFF(raw):
    raw = raw.replace(b'\xff\x00',b'\xff')
    return raw

def get_boundary(res):
    i = 0
    while i < len(res):
        if res[i]==chr(255) and res[i+1]!=chr(0):
            break
        i = i+1
    return i

def get_scantable_boundary(res):
    i = 0
    while i < len(res):
        if res[i]==chr(255) and res[i+1]!=chr(0) and \
           not(ord(res[i+1])>=0xD0 and ord(res[i+1])<=0xD7):
            break
        i = i+1
    return i

def restart_boundary(res):
    i = 0
    while i < len(res):
        if res[i]==chr(255) and ord(res[i+1])>=0xD0 and ord(res[i+1])<=0xD7:
            break
        i = i+1
    return i,ord(res[i+1])-0xD0


class ComponentIterationPlan:
    def __init__(self, scanHeader, CSParams, X, Y, restart_interval=0):
        self.maxH = max( [v['Hi'] for k,v in list(CSParams.items())] )
        self.maxV = max( [v['Vi'] for k,v in list(CSParams.items())] )
        self.Ns = scanHeader['Ns']
        self.plan = []
        self.X = X
        self.Y = Y
        self.restart_interval = restart_interval
        self.restart_marker = 0

        rCount = 0
        restart = False
        marker = self.restart_marker
        #logger.debug("CIP: Ns={} restart_interval={} marker={}".format(self.Ns,self.restart_interval,self.restart_marker))

        if self.Ns == 1:
            # this is a simple top-left to bottom-right scan
            #
            assert len(scanHeader['CSParams']) == 1
            component = scanHeader['CSParams'][0][0]
            Hi = CSParams[component]['Hi']
            Vi = CSParams[component]['Vi']
            self.adjX = nearest(X*Hi//self.maxH,8)//8
            self.adjY = nearest(Y*Vi//self.maxV,8)//8            
            #self.adjX = nearest(X,8)/8*Hi/self.maxH
            #self.adjY = nearest(Y,8)/8*Vi/self.maxV
            for i in range(self.adjY):
                for j in range(self.adjX):
                    if self.restart_interval != 0:
                        if rCount == self.restart_interval:
                            #logger.debug("CIP: Found a restart marker. i={} j={}".format(i,j))
                            restart = True
                            marker = self.restart_marker
                            rCount = 0
                            self.restart_marker = (self.restart_marker+1)%8
                        else:
                            restart = False
                            marker = -1
                        rCount += 1                                                
                    self.plan.append( [component,j,i,scanHeader['CSParams'][0][1],scanHeader['CSParams'][0][2],restart,marker] )            
        else:
            self.adjY = nearest(Y,8)/8
            self.adjX = nearest(X,8)/8
            for y in range(0,self.adjY,self.maxV):
                for x in range(0,self.adjX,self.maxH):
                    if self.restart_interval != 0:
                        if rCount == self.restart_interval:
                            restart = True
                            marker = self.restart_marker
                            rCount = 0
                            self.restart_marker = (self.restart_marker+1)%8
                        else:
                            restart = False
                            marker = -1
                        rCount += 1
                                                                    
                    for j,dc,ac in scanHeader['CSParams']:
                        #print dc,ac
                        y_factor = self.maxV / CSParams[j]['Vi']
                        x_factor = self.maxH / CSParams[j]['Hi']
                        for yy in range(CSParams[j]['Vi']):
                            for xx in range(CSParams[j]['Hi']):
                                self.plan.append( [j,x/x_factor+xx,y/y_factor+yy,dc,ac,restart,marker] )
                                restart = False
        
        self.index = 0

    def get(self):
        return self.plan[self.index]

    def reset(self):
        self.index = 0
    
    def __next__(self):
        self.index += 1

    def done(self):
        return self.index == len(self.plan)
    
    def skip_to_restart(self):
        #logger.debug("skip to restart!")
        while not self.index+1 == len(self.plan) and self.plan[self.index+1][5] != True :
            self.index += 1
        #logger.debug("skip to restart! index={}".format(self.index))
    
###
### Handle Sequential and Baseline DCT decoding
###
def SOS_sequential_scan(self,data):
    sh = scan_header(self,data)
    assert sh['Ss']==0 and sh['Se']==63
    assert sh['Ah']==0 and sh['Al']==0
    data = data[sh['Ls']:]

    bound = get_boundary(data)
    reg = data[:bound]
    data = data[bound:]
    reg = removePaddingAfterFF(reg)

    b = bitarray()
    b.frombytes(reg)

    # Still some bugs in here
    adjY = nearest(self.Y,8)//8
    adjX = nearest(self.X,8)//8
    X=0
    Y=0

    lastDC = [0,0,0,0]

    #k = 0
    done = False
    while not done:
        #print y,x
        for j,dc,ac in sh['CSParams']:
            #print dc,ac
            dc_ht = self.dc_hts[dc]
            ac_ht = self.ac_hts[ac]
            y_factor = self.maxV // self.CSParams[j]['Vi']
            x_factor = self.maxH // self.CSParams[j]['Hi']
            for yy in range(self.CSParams[j]['Vi']):
                for xx in range(self.CSParams[j]['Hi']):
                    offset,block = entropy_decode_block(dc_ht,ac_ht,b)
                    if offset==0:
                        done = True
                        break
                    assert len(block)==64

                    # if X==0 and Y==0 and j==1 and xx==0 and yy==0:
                    #     print "lastDC={}".format(lastDC)
                    #     print "block = {}".format(block[0:5])
                    # if X==0 and Y==0 and j==1 and xx==1 and yy==0:
                    #     print "lastDC={}".format(lastDC)
                    #     print "block = {}".format(block[0:5])
                        
                    b = b[offset:]
                    lastDC[j] = block[0]+lastDC[j]
                    block[0] = lastDC[j]

                    # if X==0 and Y==0 and j==1 and xx==0 and yy==0:
                    #     print block
                    #     bxxx = unflatten_by_zig_zag(block)
                    #     # print "unflatten={}".format(bxxx)
                    #     # print "idct2={}".format(idct2(bxxx,self.qts[self.CSParams[j]['q'])]))
                    block = unflatten_and_IDCT_block(block,\
                                                 self.qts[self.CSParams[j]['q']])
                    
                    # if X==0 and Y==0 and j==1 and xx==0 and yy==0:
                    #     print "full call={}".format(block)
                    #print X/self.CSParams[j]['rX']+xx,Y/self.CSParams[j]['rY']+yy,block
                    #block = np.clip(block,0,255)

                    self.comps[j].set_block(X/self.CSParams[j]['rX']+xx,
                                            Y/self.CSParams[j]['rY']+yy,block)

                    #self.comps[j][Y/self.CSParams[j]['rY']+yy][X/self.CSParams[j]['rX']+xx] = block
                if done:
                    break
            if done:
                break

        X+=self.maxH       
        if X==adjX:
            X=0
            Y+=self.maxV
            
    return data


###
### Handle Progressive DCT scan decoding
###
def SOS_progressive_scan(self,data):
    sh = scan_header(self,data)
    data = data[sh['Ls']:]

    #if sh['Ah']==0:
    #    for id,dc,ac in sh['CSParams']:
            #if self.spectral_selection[id][0] == False:
            #    assert sh['Ss'] == 0 and sh['Se'] == 0
            #for i in range( sh['Ss'], sh['Se']+1, 1 ):
            #    assert self.spectral_selection[id][i] == False
    #        self.spectral_selection[id][ sh['Ss']:sh['Se']+1 ] = [ True for _ in range(sh['Ss'],sh['Se']+1,1) ]
    #else:
        #for id,dc,ac in sh['CSParams']:
            #for i in range( sh['Ss'], sh['Se']+1, 1 ):
            #    assert self.spectral_selection[id][i] == True                    
    if sh['Ss']==0:
        if sh['Ah'] == 0:
            return SOS_progressive_DC_scan(self,sh,data)
        else:
            return SOS_progressive_DC_successive_scan(self,sh,data)
    else:
        if sh['Ah'] == 0:
            return SOS_progressive_AC_first_scan(self,sh,data)
        else:
            return SOS_progressive_AC_successive_scan(self,sh,data)

def progressive_entropy_decode_DC(dcht,b):
    ht = dcht
    l,RRRRSSSS = ht.decode(get_bits_to_decode(b[0:16]))
    if l==0:
        return l,None
    z = RRRRSSSS//16
    m = RRRRSSSS & 0xF
    val = b[l:l+m]
    assert z==0
    assert m!=15
    if m==0:
        DC_val = 0
    else:
        DC_val = decode_low_bits(val,m)
    return l+m,DC_val

###
### Handle DC coefficients for Progressive DCT decoding
###
def SOS_progressive_DC_scan(self,sh,data):

    #logger.debug("Progressive DC scan {}".format(sh))
    
    #print self.CSParams
    #print sh
    bound = get_boundary(data)
    reg = data[:bound]
    data = data[bound:]
    reg = removePaddingAfterFF(reg)

    b = bitarray()
    b.frombytes(reg)

    adjY = nearest(nearest(self.Y,8)/8,self.maxV)
    adjX = nearest(nearest(self.X,8)/8,self.maxH)
    X=0
    Y=0
    lastDC = [0,0,0,0]

    cip = ComponentIterationPlan(sh,self.CSParams,self.X,self.Y,self.restart_interval)
    #logger.debug("Build iteration plan.")

    done = False
    while not cip.done():
        #print y,x
        j,x,y,dc,ac,restart,rmarker = cip.get()
        #logger.debug("j={} x={} y={} restart={} marker={}".format(j,x,y,restart,rmarker))
        dc_ht = self.dc_hts[dc]
        ac_ht = self.ac_hts[ac]

        if restart:
            #logger.debug("Restart is true!")
            #logger.debug("remaining bits={}".format(b.to01()))    
            lastDC = [0,0,0,0]
            unpack = [[16,0]]
            byte_unpacker(data,unpack)
            marker = unpack[0][1]

            if marker < 0xFFD0 or marker > 0xFFD7:
                logger.debug("Didn't get a marker as expected! 0x{:x}".format(marker))
                # we didn't find a restart marker! that's a problem!
                # don't know how to handle it, give up
                return data

            #logger.debug("Got a restart marker! before j = {} marker = 0x{:x}".format(j,marker))
            # got a restart marker, so advance beyond that marker
            data = data[2:]

            if marker == 0xFFD0+rmarker:
                logger.debug("Got marker that was expected!")
                # got what expected!
                pass
            else:
                #logger.debug("Did NOT get what was expected = {:x}!".format(marker))
                # did not get the marker we expected!
                # Skip ahead and find the next place to start decoding,
                # but if we don't find a suitable place, give up
                while not cip.done():
                    j,x,y,dc,ac,restart,rmarker = cip.get()
                    if restart and rmarker == marker:
                        break
                    next(cip)

                if cip.done():
                    # this means we never reconverged
                    return data
            
            bound = get_boundary(data)
            reg = data[:bound]
            data = data[bound:]
            b = bitarray()
            reg = removePaddingAfterFF(reg)
            b.frombytes(reg)

            # get next interval of data
            # expect to find restart marker!
            # if we don't, it's an error, and scan ahead until we find it


        offset,DC_val = progressive_entropy_decode_DC(dc_ht,b)
        if offset==0:
            # A few possibilities:
            #  1. Missing data causing corrupt scan.
            #  2. (same as 1) We could be missing data and immediately find a restart marker.
            #  3. It's really the end (we can easily check this)
            
            # Handle case 3 first:
            if cip.done()==True:
                #logger.debug("Got to end of scan.")
                break
            else:
                logger.debug("Failed to get data. Try to skip ahead to handle restart process.")
                # skip ahead to next restart and process it
                next(cip)
                continue
        
        b = b[offset:]
        lastDC[j] = DC_val+lastDC[j]
        DC_val = lastDC[j]
        #logger.debug("x={} y={} dc_val={}".format(x,y,DC_val))
        DC_val = DC_val << (sh['Al']) # if Al==0, this is multiply by 1                       
        self.comps[j].set_spectral(x,y,0,DC_val,True)
        next(cip)
            
    assert cip.done() == True
    #logger.debug("remaining bits={}".format(b.to01()))    
    return data


###
### Handle DC coefficients for Progressive DCT decoding
###
def SOS_progressive_DC_successive_scan(self,sh,data):

    #print self.CSParams
    #print sh
    bound = get_boundary(data)

    
    reg = data[:bound]
    data = data[bound:]
    reg = removePaddingAfterFF(reg)

    #print "bound={}".format(len(reg))

    #return data
    b = bitarray()
    b.frombytes(reg)

    adjY = nearest(nearest(self.Y,8)/8,self.maxV)
    adjX = nearest(nearest(self.X,8)/8,self.maxH)
    X=0
    Y=0

    #FIXME: calculate CSParams based on actual components present
    #in this scan. May include 1, 2, or 3 components

    #k = 0
    done = False
    i = 0
    while not done:
        #print y,x
        for j,dc,ac in sh['CSParams']:
            #print dc,ac
            dc_ht = self.dc_hts[dc]
            ac_ht = self.ac_hts[ac]
            y_factor = self.maxV / self.CSParams[j]['Vi']
            x_factor = self.maxH / self.CSParams[j]['Hi']
            for yy in range(self.CSParams[j]['Vi']):
                for xx in range(self.CSParams[j]['Hi']):
                    #print X/x_factor,Y/y_factor,j,dc,ac,xx,yy

                    val = int(b[i])
                    i+=1

                    p_val = self.comps[j].get_spectral(X/self.CSParams[j]['rX']+xx,Y/self.CSParams[j]['rY']+yy,0,True)
                    #if int(p_val) != 0:
                    n_val = int(p_val) | (val << (sh['Al']))
                    #else:
                    #    # FIXME not sure about this
                    #    if val==1:
                    #        n_val = (1 << (sh['Al']))
                    #    else:
                    #        n_val = -(1 << (sh['Al']))
                    
                    #print j,Y/self.CSParams[j]['rY']+yy,X/self.CSParams[j]['rX']+xx

                    #print "j={} x={} y={} p_val={} new_val={}".format(j,X+xx,Y+yy,\
                    #    p_val,\
                    #                                                  n_val)

                    #p_val = successive_update(p_val,sh['Al'],val)
                    
                    self.comps[j].set_spectral(X/self.CSParams[j]['rX']+xx,Y/self.CSParams[j]['rY']+yy,0,n_val,True)
                    #self.comps[j][][][0] = DC_val
                if done:
                    break
            if done:
                break

        X+=self.maxH       
        if X==adjX:
            X=0
            Y+=self.maxV
            if Y==adjY:
                break

    #print b[i:]
    return data



def decode_EOBRUN(bits,m):
    if m==0:
        return 1
    else:
        low = _get_int_from_bits(bits)
        low += 2**m
        return low


###
### Handle AC coefficients for Progressive DCT decoding
###
def SOS_progressive_AC_first_scan(self,sh,data):
    #logger.debug("Progressive AC scan")
    
    assert sh['Ns'] == 1

    Ss = sh['Ss']
    Se = sh['Se']
    
    bound = get_boundary(data)
    reg = data[:bound]
    data = data[bound:]
    reg = removePaddingAfterFF(reg)

    b = bitarray()
    b.frombytes(reg)

    cip = ComponentIterationPlan(sh,self.CSParams,self.X,self.Y,self.restart_interval)
    #logger.debug("Build iteration plan.")

    while not cip.done():
        #print y,x
        j,X,Y,dc,ac,restart,rmarker = cip.get()
        #logger.debug("j={} x={} y={} restart={} marker={}".format(j,X,Y,restart,rmarker))
        dc_ht = self.dc_hts[dc]
        ac_ht = self.ac_hts[ac]

        if ac_ht == None:
            ac_ht = ErrorWhileDecodingTable(2,['0','1'])
        
        if restart:
            #logger.debug("Restart is true!")
            #logger.debug("remaining bits={}".format(b.to01()))    
            lastDC = [0,0,0,0]
            unpack = [[16,0]]
            byte_unpacker(data,unpack)
            marker = unpack[0][1]

            if marker < 0xFFD0 or marker > 0xFFD7:
                logger.debug("Didn't get a marker as expected! 0x{:x}".format(marker)) 
                # we didn't find a restart marker! that's a problem!
                # don't know how to handle it, give up
                return data

            #logger.debug("Got a restart marker! before j = {} marker = 0x{:x}".format(j,marker))
            # got a restart marker, so advance beyond that marker
            data = data[2:]

            if marker == 0xFFD0+rmarker:
                # logger.debug("Got marker that was expected!")
                # got what expected!
                pass
            else:
                # did not get the marker we expected!
                # Skip ahead and find the next place to start decoding,
                # but if we don't find a suitable place, give up
                while not cip.done():
                    j,x,y,dc,ac,restart,rmarker = cip.get()
                    if restart and rmarker == marker:
                        break
                    next(cip)

                if cip.done():
                    # this means we never reconverged
                    return data
            
            bound = get_boundary(data)
            reg = data[:bound]
            data = data[bound:]
            b = bitarray()
            reg = removePaddingAfterFF(reg)
            b.frombytes(reg)
        
        K = Ss - 1
        while K < Se:
            K += 1
            l,RRRRSSSS = ac_ht.decode(get_bits_to_decode(b))
            if l==0:
                done = True
                break
            m = RRRRSSSS & 0xF
            z = RRRRSSSS / 16

            if z == 0 and m == 0:
                b = b[l:]
                while K <= Se:
                    self.comps[j].set_spectral(X,Y,K,0)
                    K += 1
            elif z == 15 and m == 0:
                b = b[l:]
                for _ in range(16):
                    self.comps[j].set_spectral(X,Y,K+_,0)            
                K += 15
            elif z >=0 and z <= 14 and m == 0:
                val = b[l:l+z]
                b = b[l+z:]
                eobrun = decode_EOBRUN(val,z)
                while K <= Se:
                    self.comps[j].set_spectral(X,Y,K,0)
                    K += 1
                eobrun -= 1
                while eobrun > 0:
                    next(cip)
                    if cip.done():
                        #logger.debug("Unexpected end of image. Skip ahead to next marker.")
                        #logger.debug("Remaining bits at end of scan is {}".format(b.to01()))
                        return data
                    j,X,Y,dc,ac,restart,rmarker = cip.get()
                    #logger.debug("(eobrun={}) j={} x={} y={} restart={} marker={}".format(eobrun,j,X,Y,restart,rmarker))
                    for _ in range(Ss,Se+1):
                        self.comps[j].set_spectral(X,Y,_,0)
                    eobrun -= 1
            else:
                #if j==1 and X==0 and Y==0:
                    #print "mag z={} m={}".format(z,m)
                assert m > 0
                val = b[l:l+m]
                b = b[l+m:]
                while z > 0:
                    self.comps[j].set_spectral(X,Y,K,0)
                    K += 1
                    z -= 1

                #assert K <= Se # use to assert, don't do this instead, use it
                # to detect missing data
                if not (K <= Se):
                    # sign that something is wrong
                    cip.skip_to_restart()
                    break
                else:
                    self.comps[j].set_spectral(X,Y,K,decode_low_bits(val,m) << sh['Al'])

         
        next(cip)

    #logger.debug("Remaining bits at end of scan is {}".format(b.to01()))
    return data

            
###
### Handle AC coefficients for Progressive DCT decoding
###
def SOS_progressive_AC_first_scan_old(self,sh,data):
    assert sh['Ns'] == 1

    Ss = sh['Ss']
    Se = sh['Se']
    
    bound = get_boundary(data)
    reg = data[:bound]
    data = data[bound:]
    reg = removePaddingAfterFF(reg)

    b = bitarray()
    b.frombytes(reg)

    X=0
    Y=0

    #k = 0
    done = False

    j = sh['CSParams'][0][0]
    ac = sh['CSParams'][0][2]
    ht = self.ac_hts[ac]
    #y_factor = self.maxV / self.CSParams[j]['Vi']
    #x_factor = self.maxH / self.CSParams[j]['Hi']

    adjY = self.CSParams[j]['Y'] #nearest(self.Y,8)/8/y_factor
    adjX = self.CSParams[j]['X'] #nearest(self.X,8)/8/x_factor

    #print self.CSParams
    #print "SOS_progressive_AC_first_scan = {}".format(sh)

    K = Ss - 1
    while True:
        # get next band
        K += 1
        #print j,K,X,Y #,sh['Al'],sh['Ah']
        l,RRRRSSSS = ht.decode(get_bits_to_decode(b))
        if l==0:
            done = True
            break
        m = RRRRSSSS & 0xF
        z = RRRRSSSS / 16

        #assert type(self.comps[j][Y][X]) != type(None)

        if z == 0 and m == 0:
            #if j==1 and X==0 and Y==0:
            #    print "EOB"
            b = b[l:]
            while K <= Se:
                self.comps[j].set_spectral(X,Y,K,0)
                K += 1
        elif z == 15 and m == 0:
            #if j==1 and X==0 and Y==0:
            #    print "ZRL"
            b = b[l:]
            for _ in range(16):
                self.comps[j].set_spectral(X,Y,K+_,0)            
            K += 15
        elif z >=0 and z <= 14 and m == 0:
            val = b[l:l+z]
            b = b[l+z:]
            eobrun = decode_EOBRUN(val,z)
            while K <= Se:
                self.comps[j].set_spectral(X,Y,K,0)
                K += 1
            eobrun -= 1
            while eobrun > 0:
                X += 1
                if X==adjX:
                    Y += 1
                    X = 0
                for _ in range(Ss,Se+1):
                    self.comps[j].set_spectral(X,Y,_,0)
                eobrun -= 1
        else:
            #if j==1 and X==0 and Y==0:
                #print "mag z={} m={}".format(z,m)
            assert m > 0
            val = b[l:l+m]
            b = b[l+m:]
            while z > 0:
                self.comps[j].set_spectral(X,Y,K,0)
                K += 1
                z -= 1
            assert K <= Se
            self.comps[j].set_spectral(X,Y,K,decode_low_bits(val,m) << sh['Al'])
                                   
        if K>=Se:
            K = Ss-1
            X+=1
            if X==adjX:                
                X = 0
                Y += 1

    #print X, Y
    #assert X==0 and Y==adjY
    #logger.debug("Remaining bits at end of scan is {}".format(b.to01()))
    return data


def successive_update(pval, al, bit):
    res = 0
    if pval > 0:
        res = pval + (bit<<al)
    else:
        res = pval - (bit<<al)
    #assert res!=0
    return res

###
### Handle AC coefficients for Progressive DCT decoding
###
def SOS_progressive_AC_successive_scan(self,sh,data):
    assert sh['Ns'] == 1
    Ss = sh['Ss']
    Se = sh['Se']    
    bound = get_boundary(data)
    reg = data[:bound]
    data = data[bound:]
    reg = removePaddingAfterFF(reg)

    #return data
    b = bitarray()
    b.frombytes(reg)

    X=0
    Y=0

    #k = 0
    done = False

    j = sh['CSParams'][0][0]
    ac = sh['CSParams'][0][2]
    ht = self.ac_hts[ac]
    e,d = ht.get_tables()
    #print e
    y_factor = self.maxV / self.CSParams[j]['Vi']
    x_factor = self.maxH / self.CSParams[j]['Hi']

    adjY = self.CSParams[j]['Y'] #nearest(self.Y,8)/8/y_factor
    adjX = self.CSParams[j]['X'] #nearest(self.X,8)/8/x_factor

    #print self.CSParams
    #print sh

    K = Ss - 1
    while True:
        # get next band
        K += 1
        #print j,K,X,Y #,sh['Al'],sh['Ah']
        l,RRRRSSSS = ht.decode(get_bits_to_decode(b))
        if l==0:
            done = True
            break
        m = RRRRSSSS & 0xF
        z = RRRRSSSS / 16

        assert m==0 or m==1

        block = self.comps[j].get_block(X,Y)

        #print block
        #print "{} cw={} m={} z={} X={} Y={} K={}".format(b.length(),b[:l].to01(),m,z,X,Y,K)

        #assert type(self.comps[j][Y][X]) != type(None)
        if z == 0 and m == 0:
            #print "EOB"
            cnt_nz = 0
            while K <= Se:
                #self.comps[j][Y][X][K] = 0
                p_val = int(self.comps[j].get_spectral(X,Y,K))
                if p_val != 0:
                    val = int(b[l+cnt_nz])
                    cnt_nz += 1
                    pp_val = successive_update(p_val,sh['Al'],val)
                    #pp_val = p_val | (val << sh['Al'])
                    #print "val={} p_val={} pp_val={}".format(val,p_val,pp_val)
                    self.comps[j].set_spectral(X,Y,K,pp_val)                    
                K += 1
            b = b[l+cnt_nz:]                        
                
        elif z == 15 and m == 0:
            #print "ZRL"
            cnt_nz = 0
            zz = 16
            while zz > 0:
                p_val = int(self.comps[j].get_spectral(X,Y,K))
                if p_val != 0:
                    #assert False and "ZRL"
                    val = int(b[l+cnt_nz])
                    cnt_nz += 1
                    #p_val = p_val | (val << sh['Al'])
                    p_val = successive_update(p_val,sh['Al'],val)
                    self.comps[j].set_spectral(X,Y,K,p_val)                    
                else:
                    zz -= 1
                K += 1

            K -= 1 #CHECK THIS LATER!
            b = b[l+cnt_nz:]
            #print "ZRL after K={} cnt_nz={}".format(K,cnt_nz)
            
        elif z >=0 and z <= 14 and m == 0:
            val = b[l:l+z]
            b = b[l+z:]
            cnt_nz = 0
            eobrun = decode_EOBRUN(val,z)
            #print "EOBRUN={}".format(eobrun)
            while K <= Se:
                p_val = int(self.comps[j].get_spectral(X,Y,K))
                if p_val != 0:
                    #assert False and "EOBRUN"
                    val = int(b[cnt_nz])
                    cnt_nz += 1
                    pp_val = successive_update(p_val,sh['Al'],val)
                    #p_val = p_val | (val << sh['Al'])
                    self.comps[j].set_spectral(X,Y,K,p_val)                    
                K += 1
            eobrun -= 1
            while eobrun > 0:
                X += 1
                if X==adjX:
                    Y += 1
                    X = 0
                for K in range(Ss,Se+1):
                    p_val = int(self.comps[j].get_spectral(X,Y,K))
                    if p_val != 0:
                        #assert False and "EOBRUN"
                        val = int(b[cnt_nz])
                        cnt_nz += 1
                        p_val = successive_update(p_val,sh['Al'],val)
                        #p_val = p_val | (val << sh['Al'])
                        self.comps[j].set_spectral(X,Y,K,p_val)                    
                eobrun -= 1
            b = b[cnt_nz:]
            #print "EOBRUN after K={} cnt_nz={}".format(K,cnt_nz)
        else:
            assert m == 1
            val = b[l:l+m]
            b = b[l+m:]
            cnt_nz = 0
            #print "X={} Y={} before z={} K={}".format(X,Y,z,K)
            while z > 0 or int(self.comps[j].get_spectral(X,Y,K)) != 0:
                p_val = int(self.comps[j].get_spectral(X,Y,K))
                if p_val != 0:
                    #assert False and "m>0"
                    bval = int(b[cnt_nz])
                    cnt_nz += 1
                    p2_val = successive_update(p_val,sh['Al'],bval)
                    #if p_val < 0:
                    #    p2_val = (p_val-(bval<<sh['Al']))
                    #    #print "p2_val={} p_val={} bval={}".format(p2_val,p_val,bval)
                    #    assert p2_val!=0
                    #else:
                    #    p2_val = p_val+(bval<<sh['Al'])
                    #    #print "p2_val={} p_val={} bval={}".format(p2_val,p_val,bval)
                    #    assert p2_val!=0
                    #p2_val = p_val | (bval << sh['Al'])
                    self.comps[j].set_spectral(X,Y,K,p2_val)                    
                else:
                    z -= 1
                K += 1
            b = b[cnt_nz:]
            #print "after K={} cnt_nz={}".format(K,cnt_nz)
            assert K <= Se+1
            low = decode_low_bits(val,m)
            assert low==-1 or low==1
            self.comps[j].set_spectral(X,Y,K, low << sh['Al'])


        if K>=Se:
            # if j==2 and X==0 and Y==0:
            #     block = self.comps[j].get_block(0,0)
            #     print "X={},Y={}:{}".format(X,Y,block)                
            #     print unflatten_by_zig_zag(block)
            #     print unflatten_and_IDCT_block(block,self.qts[ self.CSParams[j]['q'] ])
            K = Ss-1
            X+=1
            if X==adjX:
                X = 0
                Y += 1

    return data



def SOS(self, b):
    if self.scan != SOS:
        return self.scan(self,b)
    else:
        unpack = [[16,0]]
        byte_unpacker(b,unpack)
        b = b[unpack[0][1]:]
        i = 0
        while i < len(b):
            if b[i]==chr(255) and b[i+1]!=chr(0):
                if b[i+1] < chr(0xD0) or b[i+1] > chr(0xD7):
                    break
            i = i+1    
        return b[i:]
    
def Skip(self, b):
    #print "Skip"
    unpack = [[16,0]]
    byte_unpacker(b,unpack)
    #print unpack,[ord(x) for x in b[:unpack[0][1]]]
    return b[unpack[0][1]:]


def DRI(self, b):
    unpack = [[16,0],[16,0]]
    byte_unpacker(b,unpack)
    #print unpack,[ord(x) for x in b[:unpack[0][1]]]
    self.restart_interval = unpack[1][1]
    logger.debug("restart_interval = {}".format(self.restart_interval))
    return b[unpack[0][1]:]
    

def DHT(self, data):
    #logger.debug("next three bytes: {}".format([ord(x) for x in data[0:3]]))
    
    table_order = [ 'Lh', 'Tc', 'Th' ]
    th = { 'DHT':16, 'Lh':16, 'Tc':4, 'Th':4 }
    unpack = [ [th['Lh'],0],
               [th['Tc'],0],
               [th['Th'],0] ]
    un = byte_unpacker(data,unpack)
    huff ={}
    for to,u in zip(table_order,un):
        huff[to] = u[1]    

    #print huff['Lh'],len(data)
    tab = JPEGHuffmanTable.decode_table(data[3:huff['Lh']])
    #print [ord(x) for x in data[3:huff['Lh']]]
    #print tab.get_raw_table()

    if huff['Tc']==0:        
        self.dc_hts[ huff['Th'] ] = tab
    else:
        #logger.debug("Register new Tc={} Th={}".format(huff['Tc'],huff['Th']))
        self.ac_hts[ huff['Th'] ] = tab
    return data[ huff['Lh']: ]
    
def DQT(self, b):
    table_order = [ 'Lq', 'Pq', 'Tq' ]
    th = { 'DQT':16, 'Lq':16, 'Pq':4, 'Tq':4 }
    unpack = [[th['Lq'],0],
              [th['Pq'],0],
              [th['Tq'],0]]
    un = byte_unpacker(b,unpack)
    qt ={}
    for to,u in zip(table_order,un):
        qt[to] = u[1]

    #print qt
        
    if qt['Pq']==0:
        self.qts[ qt['Tq'] ] = np.array([x for x in b[3:3+64]]).reshape((8,8))
    elif qt['Pq']==1:
        unpack = [ [16,0] for i in range(64) ]
        byte_unpacker(b[3:qt['Lq']],unpack)
        vals = [ u[1] for u in unpack ] 
        self.qts[ qt['Tq'] ] = np.array(vals).reshape((8,8))
    else:
        assert 0 and "got illegal DQT table entry size, should be 0 or 1"
    
    return b[qt['Lq']:]

def APP(self, data):
    table_order = [ 'La', ]
    th = { 'APP':16, 'La':16 }
    unpack = [[th['La'],0]]
    un = byte_unpacker(data,unpack)
    appt ={}
    for to,u in zip(table_order,un):
        appt[to] = u[1]
    #print appt
    appt['table'] = [x for x in data[2:appt['La']]]    
    return data[unpack[0][1]:]

def COM(self, b):
    unpack = [[16,0]]
    byte_unpacker(b,unpack)
    return b[unpack[0][1]:]

MARKER = {
    0xFFC0: ("SOF0", "Baseline DCT", SOF0),
    0xFFC1: ("SOF1", "Extended Sequential DCT", SOF1),
    0xFFC2: ("SOF2", "Progressive DCT", SOF2),
    0xFFC3: ("SOF3", "Spatial lossless", SOF),
    0xFFC4: ("DHT", "Define Huffman table", DHT),
    0xFFC5: ("SOF5", "Differential sequential DCT", SOF),
    0xFFC6: ("SOF6", "Differential progressive DCT", SOF),
    0xFFC7: ("SOF7", "Differential spatial", SOF),
    0xFFC8: ("JPG", "Extension",Skip),
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
    0xFFDA: ("SOS", "Start of scan", SOS),
    0xFFDB: ("DQT", "Define quantization table", DQT),
    0xFFDC: ("DNL", "Define number of lines", Skip),
    0xFFDD: ("DRI", "Define restart interval", DRI),
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
    0xFFF0: ("JPG0", "Extension 0", Skip),
    0xFFF1: ("JPG1", "Extension 1", Skip),
    0xFFF2: ("JPG2", "Extension 2", Skip),
    0xFFF3: ("JPG3", "Extension 3", Skip),
    0xFFF4: ("JPG4", "Extension 4", Skip),
    0xFFF5: ("JPG5", "Extension 5", Skip),
    0xFFF6: ("JPG6", "Extension 6", Skip),
    0xFFF7: ("JPG7", "Extension 7", Skip),
    0xFFF8: ("JPG8", "Extension 8", Skip),
    0xFFF9: ("JPG9", "Extension 9", Skip),
    0xFFFA: ("JPG10", "Extension 10", Skip),
    0xFFFB: ("JPG11", "Extension 11", Skip),
    0xFFFC: ("JPG12", "Extension 12", Skip),
    0xFFFD: ("JPG13", "Extension 13", Skip),
    0xFFFE: ("COM", "Comment", COM)
}

ERRORS = {
    -1: "image buffer overrun error",
    -2: "decoding error",
    -3: "unknown error",
    -8: "bad configuration",
    -9: "out of memory error"
}

class JPEGDecoder:
    def __init__(self,tolerate_errors=False):
        self.comps = [None,None,None,None]
        self.dc_hts = [None,None,None,None]
        self.ac_hts = [None,None,None,None]
        self.qts = [None,None,None,None]
        self.mode = "Unknown"
        self.tolerate_errors=tolerate_errors
        self.scan = SOS
        self.restart_interval = 0
        logger.debug("restart_interval = 0")            
        return

    def _decode(self, allbytes):
        global MARKER
        unpack = [[16, 0]]
        byte_unpacker(allbytes,unpack)
        if unpack[0][1] != 0xFFD8:
            self.jpeg = JPEG(None,1,(100,100))            
            return -2        
        allbytes = allbytes[2:]
        while True:
            unpack = [[16,0]]
            byte_unpacker(allbytes,unpack)
            logger.debug("_decode: next 6 bytes: {}".format("".join([ "{:x} ".format(_) for _ in [x for x in allbytes[0:6]]])))            
            allbytes = allbytes[2:]
            marker = unpack[0][1]
            if self.tolerate_errors == False:
                if marker in MARKER:
                    f = MARKER[marker][-1]
                    if f != None:
                        allbytes = f(self,allbytes)
                    else:
                        if MARKER[marker][0] == 'EOI':
                            break
                else:
                    logger.debug("_decode: failed to find marker 0x{:x}".format(marker))
                    loc = get_boundary(allbytes)
                    allbytes = allbytes[loc:]
                    if len(allbytes)==0:
                        return -2
            else:
                if marker in MARKER:
                    try:
                        f = MARKER[marker][-1]
                        if f != None:
                            allbytes = f(self,allbytes)
                        else:
                            if MARKER[marker][0] == 'EOI':
                                break
                    except:
                        loc = get_scantable_boundary(allbytes)
                        allbytes = allbytes[loc:]
                        if len(allbytes)==0:
                            break

                else:
                    logger.debug("_decode: failed to find marker 0x{:x}".format(marker))
                    loc = get_scantable_boundary(allbytes)
                    allbytes = allbytes[loc:]
                    if len(allbytes)==0:
                        break
                
        try:
            new_comps = []
            for c in self.comps:
                if c!=None:
                    a = c.prepare()
                    new_comps.append(a)
                else:
                    new_comps.append(c)

            ds_Y = ( self.maxH//self.CSParams[1]['Hi'] , self.maxV//self.CSParams[1]['Vi'] ) 
            ds_Cb = ( self.maxH//self.CSParams[2]['Hi'] , self.maxV//self.CSParams[2]['Vi'] ) 
            ds_Cr = ( self.maxH//self.CSParams[3]['Hi'] , self.maxV//self.CSParams[3]['Vi'] ) 

            self.jpeg = JPEG(None,1,(self.X,self.Y))
            print (len(new_comps[1]))
            self.jpeg.set_Y_blocks(np.array(new_comps[1]),ds_Y)
            self.jpeg.set_Cb_blocks(np.array(new_comps[2]),ds_Cb)
            self.jpeg.set_Cr_blocks(np.array(new_comps[3]),ds_Cr)
            self.jpeg.merge()
        except Exception as e:
            if self.tolerate_errors==False:
                return -3
            else:
                self.jpeg = JPEG(None,1,(100,100))
                pass

        return 0

    def compareY(self,jpeg):
        Yc = self.jpeg.get_Y_blocks()
        otherY = jpeg.get_Y_blocks()
        l = []
        for i,(y,yp) in enumerate(zip(Yc,otherY)):
            #print "{} vs {}".format(np.mean(y),np.mean(yp))
            #y1 = dct2(y,np.ones((8,8)))
            #y2 = dct2(yp,np.ones((8,8)))            
            #val = np.abs(y-yp)
            #l.append( idct2(dct2(yp/4*4)) )
            print(y)
            print(dct2(y,self.qts[1]))
            #print yp
            if i==10:
                break

        #a = np.array(l)
        #print "a=",a
        #a=a.reshape((self.X,self.Y))
        #tmp = JPEG(None,1,(self.X,self.Y))
        #tmp.set_Y_blocks(a,(1,1))
        #tmp.Y.show()
        
    def decode(self,**kwargs):
        if 'filename' in kwargs:
            f = open( kwargs['filename'], 'rb')
            b = f.read()
        else:
            b = kwargs['bytes']
        return self._decode(b)


class JPEGFileRepair:
    def __init__(self):
        self.comps = [None,None,None,None]
        self.dc_hts = [None,None,None,None]
        self.ac_hts = [None,None,None,None]
        self.qts = [None,None,None,None]
        self.mode = "Unknown"
        self.scan = SOS
        return

    def _decode(self, allbytes):
        global MARKER
        unpack = [[16, 0]]
        byte_unpacker(allbytes,unpack)
        if unpack[0][1] != 0xFFD8:
            return -2        
        allbytes = allbytes[2:]
        while True:
            unpack = [[16,0]]
            byte_unpacker(allbytes,unpack)
            logger.debug("_decode: next 6 bytes: {}".format("".join([ "{:x} ".format(_) for _ in [ord(x) for x in allbytes[0:6]]])))            
            allbytes = allbytes[2:]
            marker = unpack[0][1]
            if marker in MARKER:
                f = MARKER[marker][-1]
                if f != None:
                    allbytes = f(self,allbytes)
                else:
                    if MARKER[marker][0] == 'EOI':
                        break
            else:
                return -2
            
        new_comps = []
        for c in self.comps:
            if c!=None:
                a = c.prepare()
                new_comps.append(a)
            else:
                new_comps.append(c)
            
        ds_Y = ( self.maxH/self.CSParams[1]['Hi'] , self.maxV/self.CSParams[1]['Vi'] ) 
        ds_Cb = ( self.maxH/self.CSParams[2]['Hi'] , self.maxV/self.CSParams[2]['Vi'] ) 
        ds_Cr = ( self.maxH/self.CSParams[3]['Hi'] , self.maxV/self.CSParams[3]['Vi'] ) 

        self.jpeg = JPEG(None,1,(self.X,self.Y))
        self.jpeg.set_Y_blocks(np.array(new_comps[1]),ds_Y)
        self.jpeg.set_Cb_blocks(np.array(new_comps[2]),ds_Cb)
        self.jpeg.set_Cr_blocks(np.array(new_comps[3]),ds_Cr)
        self.jpeg.merge()
        return 0

    def decode(self,**kwargs):
        if 'filename' in kwargs:
            f = open( kwargs['filename'], 'rb')
            b = f.read()
        else:
            b = kwargs['bytes']
        return self._decode(b)

    
if __name__ == "__main__":
    import sys
    from dnapreview.logger import logger
    #j = JPEG(filename=sys.argv[1])
    #j.Y.show()
    #j.Cr.show()
    codec = JPEGDecoder()
    ret = codec.decode(filename=sys.argv[1])
    codec.jpeg.show()
    #codec.compareY(j)

    #for i,c in enumerate(codec.spectral_selection):
    #    print i,c
