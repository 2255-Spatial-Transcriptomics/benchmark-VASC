from pathlib import Path
import os
import numpy as np

def load_biase_data():
    CWD = os.path.dirname(os.path.realpath(__file__))
    DATAPATH = str(Path(CWD).parent) + '/data/'
    print(DATAPATH)
    DATASET = 'biase' #sys.argv[1]
    PREFIX = 'biase' #sys.argv[2]

    filename = DATAPATH + DATASET + '.txt'
    data = open( filename )
    head = data.readline().rstrip().split()

    label_file = open( DATAPATH + DATASET+'_label.txt' )
    label_dict = {}
    for line in label_file:
        temp = line.rstrip().split()
        label_dict[temp[0]] = temp[1]
    label_file.close()

    label = []
    for c in head:
        if c in label_dict.keys():
            label.append(label_dict[c])
        else:
            print(c)

    label_set = []
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value:idx for idx,value in enumerate(label_set)}
    id_map = {idx:value for idx,value in enumerate(label_set)}
    label = np.asarray( [ name_map[name] for name in label ] )
    
    expr = []
    for line in data:
        temp = line.rstrip().split()[1:]
        temp = [ float(x) for x in temp]
        expr.append( temp )
    
    expr = np.asarray(expr).T
    
    return expr, label, PREFIX