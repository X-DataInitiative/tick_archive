#!/bin/bash

rm -rf build_noopt build_mkl build_omp build_omp_mkl
mkdir build_noopt build_mkl build_omp build_omp_mkl

CMAKE_DIR='../..'

echo ${CMAKE_DIR}

(cd build_noopt && cmake ${CMAKE_DIR})
(cd build_mkl && cmake -DUSE_MKL=ON ${CMAKE_DIR})
(cd build_omp && cmake -DUSE_OPENMP=ON ${CMAKE_DIR})
(cd build_omp_mkl && cmake -DUSE_MKL=ON -DUSE_OPENMP=ON ${CMAKE_DIR})
