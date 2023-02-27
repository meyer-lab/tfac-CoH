SHELL := /bin/bash

flist = 1 2 3 4 5 S1 S2 S3 S4 S5

.PHONY: clean test all

all: $(patsubst %, output/figure%.svg, $(flist))

output/figure%.svg: coh/figures/figure%.py
	mkdir -p ./output
	poetry run fbuild $*

clean:
	rm -rf output
