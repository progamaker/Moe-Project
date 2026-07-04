#!/usr/bin/env bash
# Скачивание MNIST (idx-формат) в data/. Хосты yann.lecun.com часто недоступны,
# используем зеркало на GitHub.
set -e
mkdir -p data && cd data
BASE="https://raw.githubusercontent.com/fgnt/mnist/master"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  echo "downloading $f"
  curl -sL -o "$f" "$BASE/$f"
done
gzip -t *.gz && echo "MNIST готов в ./data"
