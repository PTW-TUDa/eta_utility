#!/usr/bin/env bash

workingDir=$1
fmuName=$2

python -V
python3 -m venv ./venv
source ./venv/bin/activate
pip install fmpy

apt-get update
apt-get --yes install zip

cd "$workingDir" || exit

\[ ! -d ./fmu_extract ] && mkdir ./fmu_extract
unzip -u ./"$fmuName".fmu -d ./fmu_extract
\[ ! -d ./fmu_extract/binaries/linux64 ] && mkdir ./fmu_extract/binaries/linux64

gcc -c -I. -I ./venv/lib/python3.7/site-packages/fmpy/c-code -fPIC ./fmu_extract/sources/all.c &&
  gcc -static-libgcc -shared -o ./fmu_extract/binaries/linux64/out.so ./fmu_extract/sources/*.o

mv ./fmu_extract/sources/out.so ./fmu_extract/binaries/linux64/"$fmuName".so
zip -r /"$fmuName".fmu fmu_extract/*
rm -rf fmu_extract
