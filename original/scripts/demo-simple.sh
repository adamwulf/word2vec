#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/simpledata.txt
VECTOR_DATA=$DATA_DIR/simpledata.bin

pushd ${SRC_DIR} && make; popd

rm $VECTOR_DATA;

if [ ! -e $VECTOR_DATA ]; then
if [ ! -e $TEXT_DATA ]; then
    echo no data
    exit 1;
fi
echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1

fi

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $DATA_DIR/$VECTOR_DATA
