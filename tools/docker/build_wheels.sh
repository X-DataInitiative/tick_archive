#!/bin/sh

set -e -x

cp -R /io src
cd src

eval "$(pyenv init -)" 

case "${PYVER}" in
    py34)
        pyenv local 3.4.5
        ;;
    py35)
        pyenv local 3.5.2
        ;;
esac

python -V
python setup.py bdist_wheel

for whl in dist/*.whl; do
	python -mauditwheel repair "$whl" -w /io/dist/wheelhouse
done
