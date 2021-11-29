# image-compression
This repository contains the implementation of a JPEG-like algorithm for the task of image compression. 

The algorithm consists of the following parts:
- Converting the image from RGB to YCbCr format
- Splitting the image into patches and computing 2-D DCT2
- Quantizing the output to remove high frequency content
- To exploit sparsity and to place lower frequencies at the beginning, unrolling each patch is done using the Zig-Zag unrolling algorithm
- ZRLE is used encode the vector which consists of three values r,s,c
  - r - Number of running zeros b/w current and prev non-zero element
  - s - The number of bits required to represent the current non-zero element
  - c - The non zero element itself
- The ZRLE tuples are encoded using Huffman Tables
- The decoding process is the reverse of all the above 
