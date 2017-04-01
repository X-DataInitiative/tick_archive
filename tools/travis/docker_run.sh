#!/bin/sh

set -e -x

# Copy tick source from host to container
cp -R /io src
cd src

eval "$(pyenv init -)"

case "${PYTOK}" in
    py34)
        export TICK_PYVER="3.4.5"
        ;;
    py35)
        export TICK_PYVER="3.5.2"
        ;;
esac

pyenv global $TICK_PYVER

python setup.py cpplint build_ext --inplace cpptest pytest

export PYTHONPATH=${PYTHONPATH}:`pwd` && (cd doc && make doctest)
