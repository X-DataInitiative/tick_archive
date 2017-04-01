#!/usr/bin/env bash

set -e -x

shell_session_update() { :; }

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