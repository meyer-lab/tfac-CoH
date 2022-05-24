SHELL := /bin/bash

flist = 1

.PHONY: clean test all

all: $(patsubst %, output/figure%.svg, $(flist))

output/figure%.svg: CoH/figures/figure%.py
	mkdir -p ./output
	poetry run fbuild $*

clean:
	rm -rf output

test:
	poetry run pytest -s -v -x
