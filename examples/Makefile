# Example make commands for preview infrastructure

.PHONY: all enc view dec

all: enc view dec

# decrease size of downsample to 1 to make a true sized image
enc:
	python ../tools/preview.py --encode --downsample=20 --preview-encoding-info preview-file.csv --o 4.earth.dna earth.jpg

dec: enc
	python ../tools/preview.py --decode --primer5 CAGGTACGCAGTTAGCACTC --primer3 CGTGGCAATATGACTACGGA 4.earth.dna --o 4.jpg

view: enc
	python ../tools/preview-viewer.py --preview-encoding-info preview-file.csv 4.earth.dna

clean:
	@rm -Rf 4.earth.dna viewer-4.earth* 4.earth.dna.comments 4.jpg
	@rm -f *~
