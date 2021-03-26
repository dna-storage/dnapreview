from PIL import Image
import numpy as np

def copy_block_into_image(block,img,ds=(1,1)):
    assert img.size[0] == block.shape[0]*ds[0]
    assert img.size[1] == block.shape[1]*ds[1]
    data = []
    for j in range(0,img.size[1],ds[1]):
        for dsy in range(ds[1]):
            for i in range(0,img.size[0],ds[0]):
                # Please fix me: inefficient!
                for dsx in range(ds[0]):
                    data.append( block[j//ds[1]][i//ds[0]] )
    img.putdata(data)

class JPEG:
    def __init__(self, filename, ds=1, size=None):
        if filename != None:
            self.filename = filename
            self.image = Image.open(filename).convert('YCbCr')
            if size==None:
                size = [ s // ds for s in self.image.size ]            
            self.image = self.image.resize( size )
            #print "image size = {}".format(self.image.size)
        else:
            self.filename = None
            self.image = Image.new('YCbCr',size)
        self.Y,self.Cb,self.Cr = self.image.split()

            
    def _get_ndarray_block(self, block, x_range=8, y_range=8, x_ds=1, y_ds=1):
        arr = []
        for y in range(0,y_range,y_ds):
            array = []
            for x in range(0,x_range,x_ds):
                array.append(block.getdata().getpixel((x,y)))
            arr.append(np.array(array))
        return np.array(arr)

    def _set_blocks(self, block, img, ds=(1,1)):        
        b = 0
        for y in range(0,img.size[1],8*ds[1]):
            for x in range(0,img.size[0],8*ds[0]):
                box = img.crop((x,y,x+8*ds[0],y+8*ds[1]))
                copy_block_into_image(block[b],box,ds)                
                img.paste(box,(x,y,x+8*ds[0],y+8*ds[1]))
                b += 1

    def _get_blocks(self, img, ds=(1,1)):
        blocks = []
        for y in range(0,img.size[1],8*ds[1]):
            for x in range(0,img.size[0],8*ds[0]):
                box = img.crop((x,y,x+8*ds[0],y+8*ds[1]))
                a = self._get_ndarray_block(box,8*ds[0],8*ds[1],ds[0],ds[1])
                blocks.append(a)
        return np.array(blocks)

    def get_Y_blocks(self,ds=(1,1)):
        return self._get_blocks(self.Y,ds)

    def set_Y_blocks(self, newY, ds=(1,1)):
        self._set_blocks(newY,self.Y,ds)

    def get_Cr_blocks(self,ds=(2,2)):
        return self._get_blocks(self.Cr,ds)

    def set_Cr_blocks(self,newCr,ds=(2,2)):
        self._set_blocks(newCr,self.Cr,ds)

    def get_Cb_blocks(self,ds=(2,2)):
        return self._get_blocks(self.Cb,ds)

    def set_Cb_blocks(self,newCb,ds=(2,2)):
        self._set_blocks(newCb,self.Cb,ds)

    def merge(self):
        self.image = Image.merge('YCbCr', (self.Y,self.Cb,self.Cr))

    def show(self):
        self.image.show()
