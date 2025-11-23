#!/bin/bash

DATA_DIR="../data"
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

echo "Downloading CIFAR-10 dataset..."
mkdir -p $DATA_DIR
cd $DATA_DIR

wget -nc $CIFAR_URL
tar -xzf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* .
rm -rf cifar-10-batches-bin cifar-10-binary.tar.gz

echo "CIFAR-10 dataset downloaded and extracted to $DATA_DIR"
echo "Files:"
ls -la *.bin