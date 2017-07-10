#!/bin/bash

for d in build_noopt build_mkl build_omp build_omp_mkl; do echo ${d} && (cd ${d} && make -j8); done
