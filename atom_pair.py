import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import pandas as pd
import numpy as np

atom_pair_index = {
    '[PAD]':0,
    'C-H':1, 
    'O-H':2, 
    'C-O':3, 
    'C-Cl':4, 
    'S-O':5,  
    'O-C':6, 
    'C-N':7, 
    'N-H':8, 
    'C-C':9, 
    'S-C':10,
    'C-S':11, 
    'C-F':12,
    'N-N':13, 
    'O-O':14,
    'N-C':15
    }

def get_geom_rdkit( smi , max_try=100):
    mol = Chem.AddHs( Chem.MolFromSmiles(smi) )
    if mol is None:
        return None
    num_try= 0
    while num_try < max_try:
        try:
            AllChem.EmbedMolecule( mol ) 
            AllChem.MMFFOptimizeMolecule( mol ) 
            return mol
        except Exception as e:
            num_try+=1
            continue
    print(f'Failed to get geo max_try {max_try}: {smi}')

def get_dist( smi ):
    m = get_geom_rdkit( smi ) #type(output) = mol
    if m is None:
        return None
    dist_mat = Chem.Get3DDistanceMatrix(m)
    vocab_list = []
    dist_list  = []
    for bond in m.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i_atom, j_atom = m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(j).GetSymbol()
        dist = dist_mat[i,j]
        vocab_idx = atom_pair_index.get(f'{i_atom}-{j_atom}')
        vocab_list.append(vocab_idx)
        dist_list.append(dist)
    return torch.tensor(vocab_list).long(), torch.tensor(dist_list).float()
#    return vocab_list, dist_list

if __name__ == '__main__':
    list_smi = [line.strip() for line in open('./data/1_Pretrain/1k_pretrain.txt')]
#    output = [get_dist(smi) for idx, smi in list_smi]
#    print(output, len(output))
    idx=0
    for idx, smi in enumerate(list_smi):
        if get_dist(smi) is not None:
            idx+=1
            print(f'{idx}/{len(list_smi)}')
    print(idx)
    exit(-1)
    uniq_atom_set = set(sum(output, []))
    print(len(uniq_atom_set), uniq_atom_set)
    exit(-1)
    #for line in output:
        #print(line)
    df = pd.DataFrame(output, columns=['atom_pair', 'dist'])
    print(df)
    print(atom_pair_dict)
    vocab_idx = [ output[idx][0] for idx in range(len(output))]
    print(vocab_idx)
    vocab_idx = [ atom_pair_dict.get(output[idx][0]) for idx in range(len(output))]
    print(vocab_idx)
    df['idx'] = vocab_idx
    print(df)
    exit(-1)
    
