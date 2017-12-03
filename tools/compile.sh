#!/bin/sh
GITCOMMIT=`git describe --tags --long --always --dirty`
echo ${GITCOMMIT}
g++ -O3 -DGITCOMMIT="\"${GITCOMMIT}\"" -o htk2nc htk2nc.cpp -I/home/zchen/usr/local/include -L/home/zchen/usr/local/lib -lnetcdf #--help=optimizers
g++ -O3 -std=c++11 -DGITCOMMIT="\"${GITCOMMIT}\"" -onc-standardize nc-standardize.cpp -I/home/zchen/usr/local/include -L/home/zchen/usr/local/lib -lnetcdf -lm
#ln -s nc-standardize nc-standardize-input
