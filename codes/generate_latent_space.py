# -*- coding: utf-8 -*-
import numpy as np
import h5py
from vasc import train_vasc
from helpers import clustering,measure,print_2D
from config import config
from pathlib import Path
import os
from scipy.io import mmread
import pandas as pd

if __name__ == '__main__':
    data_dir = 'data/'
    mtx = data_dir + 'lung_human_ASK440.mtx'
    PREFIX = 'lung_human_ask440'
    x = mmread(mtx).astype('int32')
    x = x.transpose().todense()
    labels = np.loadtxt('data/lung_human_ASK440_celltype.tsv', delimiter = '\t', dtype = str)
    encoded_labels = list(pd.get_dummies(labels).columns)
    integer_labels = []
    for string in labels:
        integer_labels.append(encoded_labels.index(string))

    n_cell,_ = x.shape
    id_map = {idx:value for idx,value in enumerate(encoded_labels)}


    if n_cell > 150:
        batch_size=config['batch_size']
    else:
        batch_size=32 


    
    for l in [2, 10, 20]:
        vae, losses = train_vasc( x,var=False,
                    latent=l,
                    annealing=False,
                    batch_size=batch_size,
                    prefix=PREFIX,
                    label=integer_labels,
                    scale=config['scale'],
                    patience=config['patience'] 
                )
        
    
    

        filename = f"{PREFIX}_{l}_res.h5"

        with h5py.File(filename, "r") as f:
            # Print all root level object names (aka keys) 
            # these can be group or dataset names 
            print("Keys: %s" % f.keys())
            # get first object name/key; may or may NOT be a group
            a_group_key = list(f.keys())[1]

            # get the object type for a_group_key: usually group or dataset
            print(type(f[a_group_key])) 

            # If a_group_key is a group name, 
            # this gets the object names in the group and returns as a list
            data = list(f[a_group_key])

            # If a_group_key is a dataset name, 
            # this gets the dataset values and returns as a list
            data = list(f[a_group_key])
            # preferred methods to get dataset values:
            ds_obj = f[a_group_key]      # returns as a h5py dataset object
            ds_arr = f[a_group_key][()]  # returns as a numpy array

        latent_space = ds_arr[-1, :, :]
        np.savetxt(f"vasc_{PREFIX}_latent_space_{l}.tsv", latent_space, delimiter=" ")

