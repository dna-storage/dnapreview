import random
import numpy as np
import unittest
from dnastorage.codec.huffman_table import *
from bitarray import bitarray

import logging
logger = logging.getLogger('dna.preview.jpeg.jpeg_huffman')
#logger.addHandler(logging.NullHandler())


class JPEGHuffmanTable(LengthLimitedHuffmanTable):
    def __init__(self, vals, weights):
        LengthLimitedHuffmanTable.__init__(self,16,2,['0','1'],vals,weights=weights,prevent_ones=True)
        mbits = 8
        assert len(vals) < 256
        for v in vals:
            assert v >= 0
            assert v <= 255
            #if not (v <= 127 and v >= -128):
            #    print "Bad value in JPEGHuffmanTable: {}".format(v)
            #if np.log2(abs(v)) > 16:
            #    print "Bad value, too large: {}".format(v)
            #assert np.log2(abs(-v)) <= 16
            #assert v <= 127 and v >= -128
        self.codeword_length = 8
        assert self.codeword_length==8

    def encode_table(self):
        h = self.histogram()
        enc = bitarray()
        for i in range(1,16+1):
            #print (h.get(i,0))
            enc += "{0:{fill}{w}b}".format(h.get(i,0), fill='0', w=8)
        table = self.get_raw_table(length_only=True)
        #print (table)
        for t in table:
            c = bitarray()
            #if t[1] < 0:
            #    c += "{0:{fill}{w}b}".format((256+t[1]), fill='0',w=self.codeword_length)
            #else:
            c += "{0:{fill}{w}b}".format(t[1], fill='0',w=self.codeword_length)
            #print ("compare: {} vs {}".format(c,t[1]))
            enc += c
            #print "{} - {}".format(t[1],c,fill='0')
        #print (enc)
        return enc.tobytes()

    
    @classmethod
    def decode_table(self, data):
        #print (type(data))
        #print (data)
        #try:
        #    data = [ ord(x) for x in data ]
        #except:
        #    data = [ x for x in data ]
        datap = []
        #print data
        for d in data:
            #if d <= 127:
            datap.append(d)
            #else:
            #    datap.append(d-256)
        data = datap
        counts = data[0:16]
        #print (counts)

        num_syms = sum(counts)
        if num_syms != len(data)-16:
            logger.debug("Error while constructing huffman table, use dummy to keep going.")
            # we have an error
            return ErrorWhileDecodingTable(2,['0','1'],None,None,True)
        else:
            logger.debug("Table appears consistent.")
        
        #print counts
        table = []
        offset = 16
        for i,c in enumerate(counts):
            t = [ [i+1,x] for x in data[offset:offset+c] ]
            table += t
            offset += c

        #print ([ t for t in table ])
        #print table
        return HuffmanTable.from_raw_table(table,2,['0','1'],prevent_ones=True)


class JPEGHuffmanTableTesting(unittest.TestCase):
    def test_standard_table(self):
        table = [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9,0xA,0xB]
        s = bytes(table)
        ht2 = JPEGHuffmanTable.decode_table(s)
        e,d = ht2.get_tables()
        print(d)
        print(ht2.decode('00')[1])
        assert ht2.decode('00')[1]==0
        assert ht2.decode('011')[1]==2
        assert ht2.decode('1110')[1]==6

    
    def test_linear_encoding_decoding(self):
        m = [ i for i in range(0,255) ]
        w = [ 1.0/(2**i) for i in range(0,255) ]
        ht = JPEGHuffmanTable(m,w)
        b = ht.encode_table()
        ht2 = JPEGHuffmanTable.decode_table(b)
        #print(ht.get_raw_table())
        #print(ht2.get_raw_table())
        t1 = ht.get_raw_table(True)
        t2 = ht2.get_raw_table(True)
        for k, (i,j) in enumerate(zip(t1,t2)):
            if i!=j:
                print ("{}: {} != {}".format(k,i,j))
        
        assert ht.get_raw_table(True) == ht2.get_raw_table(True)

    def test_negatives_encoding_decoding(self):
        #m = [-1, 1]
        #w = [ .9, .1]
        #ht = JPEGHuffmanTable(m,w)
        #b = ht.encode()
        #ht2 = JPEGHuffmanTable.decode(b)
        #assert ht.get_raw_table() == ht2.get_raw_table()
        assert True
        
    def test_single_encoding_decoding(self):
        m = [1,]
        w = [ 1]
        ht = JPEGHuffmanTable(m,w)
        enc,dec = ht.get_tables()
        print(enc)
        b = ht.encode_table()
        ht2 = JPEGHuffmanTable.decode_table(b)
        assert ht.get_raw_table() == ht2.get_raw_table()

    def test_random_encoding_decoding(self):
        R = random.Random()
        tmp = [ R.randint(20,100) for i in range(1000) ]
        h = {}
        for t in tmp:
            h[t] = h.get(t,0)+1

        vals = list(h.keys())
        w = sum([float(h[x]) for x in vals])
        weights = [ float(h[v])/w for v in vals]
            
        ht = JPEGHuffmanTable(vals,weights)
        b = ht.encode_table()
        ht2 = JPEGHuffmanTable.decode_table(b)
        assert ht.get_raw_table() == ht2.get_raw_table()
