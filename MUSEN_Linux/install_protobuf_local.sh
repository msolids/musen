#!/bin/bash

# installation path
PROTO_INSTALL_PATH=$PWD/MUSEN_externals/protobuf_install

# path to zlib
ZLIB_INSTALL_PATH=$PWD/MUSEN_externals/zlib_install

# version to compile
PROTO_VER=3.9.1
DIR_NAME=protobuf-$PROTO_VER
TAR_NAME=protobuf-cpp-$PROTO_VER.tar.gz

# clear old
rm -rf $PROTO_INSTALL_PATH

# build protobuf
wget https://github.com/protocolbuffers/protobuf/releases/download/v$PROTO_VER/$TAR_NAME
tar -xvzf $TAR_NAME
cd $DIR_NAME
./configure --with-zlib --with-zlib-include=$ZLIB_INSTALL_PATH/include --with-zlib-lib=$ZLIB_INSTALL_PATH/lib --disable-examples --disable-tests --prefix=$PROTO_INSTALL_PATH
make -j8
make install

# remove unnecessary 
rm -rf $PROTO_INSTALL_PATH/lib/pkgconfig

# clean build stuff
cd ../
rm $TAR_NAME
rm -rf $DIR_NAME
