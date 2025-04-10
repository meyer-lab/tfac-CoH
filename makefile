SHELL := /bin/bash

flist = $(wildcard coh/figures/figure*.py)

.PHONY: clean test all

all: $(patsubst coh/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: coh/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

clean:
	rm -rf output
