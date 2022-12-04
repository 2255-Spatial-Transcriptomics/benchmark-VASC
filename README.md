# VASC
#### Variational autoencoder for single cell RNA-seq datasets

Single cell RNA sequencing (scRNA-seq) is a powerful technique to analyze the transcriptomic heterogeneities in single cell level. It is an important step for studying the cell sub-populations and lineages from scRNA-seq data by finding an effective low-dimensional representation and visualization of the original data. The scRNA-seq data are more “noisy” than traditional bulk RNA-seq: in the single cell level, the transcriptional fluctuations are much larger than the average of a cell population and the low amount of RNA transcripts will increase the rate of technical dropout events. In this study, we proposed VASC (deep Variational Autoencoder for SCRNA-seq data), a deep multi-layer generative model, for the dimension reduction and visualization. It can do nonlinear hierarchical feature representations and model the dropout events of scRNA-seq data. Tested on more than twenty datasets, VASC show better performances in most cases and higher stability compared with several dimension reduction methods. VASC successfully re-establishes the embryo pre-implantation cell lineage and its associated genes based on the 2D representation of a large-scale scRNA-seq from human embryos.

## Read the paper [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6364131/)

## Prerequisites
+ Python 3.5+
+ numpy 1.12.1
+ h5py 2.7.0
+ sklearn 0.18.1
+ tensorflow 1.1.0
+ keras 2.0.6

We recommend to install the newest Anaconda from https://www.continuum.io/downloads.

## Codes
Two python files are included:
- vasc.py: contains a class VASC and a function vase
- helpers.py: auxiliary functions

## Demo
We gave a demo.py and config.py to demonstrate the use of VASC.

## Data
A small dataset from Biase is included for demonstration.

## Notes for 2255 Spatial Transcriptomics

VASC uses the expression matrix from scRNA-seq data as inputs. The whole expression matrix of the transcriptome was fed directly to the model with no gene filter applied. The data were log-transformed to make the results more robust. The most important transformation, however, was to re-scale the expression of every gene in any single cell in the range [0,1] by dividing the maximum expression value of an individual gene from the same cell.

## installation notes on mac m1

Only working way to install tensorflow on macos M1 chip: https://developer.apple.com/forums/thread/683757

I was able to get this working with keras 2.10.0
changed a few lines of code 

keras.layers.merge()  to Concatente and Multiply

