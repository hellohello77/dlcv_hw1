#!/bin/bash

# TODO - run your inference Python3 code
start=$(date +%s)
# curl -L -O first87_28.ckpt 'https://docs.google.com/uc?export=download&id=1Ja-7-1HZuFvlP1D2OFqtQrDKXrVol13C'
# wget -O first87_28.ckpt https://drive.google.com/file/uc?id=1Ja-7-1HZuFvlP1D2OFqtQrDKXrVol13C
# wget -O first87_28.ckpt https://www.dropbox.com/s/2kk53fkeih1vm79/Dropbox%20%E6%96%B0%E6%89%8B%E6%8C%87%E5%8D%97.pdf?dl=1

python3 hw1_1_inf.py $1 $2
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"