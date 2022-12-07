# -*- coding: utf-8 -*-
import numpy as np
from vasc import train_vasc
from helpers import clustering,measure,print_2D
from load_data import load_biase_data
from config import config
import tensorflow as tf
import os
import keras
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    np.random.seed(2)
    expr, label, PREFIX = load_biase_data()
    
    # split into training and validation
    split_percentage = 20
    indices = np.random.choice(len(label), int(len(label)*split_percentage/100), replace=False)
    
    train_expr = np.delete(expr, indices, axis=0)
    train_labels = np.delete(label, indices)
    
    val_expr = expr[indices,:]
    val_labels = label[indices]
    
    print("validation labels: ", val_labels)

    n_cell,_ = train_expr.shape
    if n_cell > 150:
        batch_size=config['batch_size']
    else:
        batch_size=32
   
    '''
    layer: {
        train:{nmi, kl, completeness, ...}
        val:{nmi, kl, completeness, ...}
    }
    '''
    
   
    latent_metrics = {}
    latent_levels = [2,4,8,16]
    for l in latent_levels:
        print('using latent dim of ', l)
        
        vasc_model, train_loss = train_vasc(train_expr,
                    epochs = 50,
                    var=False,
                    latent=l,
                    annealing=False,
                    batch_size=batch_size,
                    prefix=PREFIX,
                    label=train_labels,
                    scale=config['scale'],
                    patience=config['patience'],
                    log=config['log'],
                )

        every_layer_pred_train = vasc_model.ae.predict([train_expr, np.ones(train_expr.shape)*1.0])
        train_reconstruction = every_layer_pred_train[-1]
        train_msmts = measure(train_expr.flatten(), train_reconstruction.flatten())

        
        every_layer_pred_val = vasc_model.ae.predict([val_expr, np.ones(val_expr.shape)*1.0])
        val_reconstruction = every_layer_pred_val[-1]
        val_msmts = measure(val_expr.flatten(), val_reconstruction.flatten())

        latent_metrics[l] = {'train':{}, 'val':{}}
        latent_metrics[l]['train'] = train_msmts
        latent_metrics[l]['val'] = val_msmts

    
    print('\n\n')
    import pprint 
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(latent_metrics)

        
    
    # if not os.path.exists("checkpoints"):
    #     os.makedirs('checkpoints')
    
    # ckpt_name = 'vasc_ae'
    # vasc_model.ae.save(f"checkpoints/{ckpt_name}")
    # print(f'checkpoint {} saved!')

    
    
    