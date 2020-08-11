#!/bin/bash

# installation path
ZLIB_INSTALL_PATH=$PWD/MUSEN_externals/zlib_install

# version to compile
ZLIB_VER=1.2.11
DIR_NAME=zlib-$ZLIB_VER
TAR_NAME=zlib-$ZLIB_VER.tar.gz

# clear old
rm -rf $ZLIB_INSTALL_PATH

# build zlib
wget http://www.zlib.net/$TAR_NAME
tar -xvzf $TAR_NAME
cd $DIR_NAME
./configure --prefix=$ZLIB_INSTALL_PATH
make -j8
make install

# remove unnecessary 
rm -rf $ZLIB_INSTALL_PATH/share
rm -rf $ZLIB_INSTALL_PATH/lib/pkgconfig

# clean build stuff
cd ../
rm $TAR_NAME
rm -rf $DIR_NAME
