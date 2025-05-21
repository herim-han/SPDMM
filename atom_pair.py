import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import pandas as pd
import pickle
from calc_property import calculate_property
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

atom_pair_index = pickle.load(open('atom_pair_vocab.pkl', 'rb') )

def get_geom_rdkit( smi , max_try=10):
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
#        print(i_atom, j_atom)
#        print(atom_pair_index.get(f'{i_atom}-{j_atom}')
        dist = dist_mat[i,j]
        vocab_idx = atom_pair_index.get(f'{i_atom}-{j_atom}')
        vocab_list.append(vocab_idx)
        dist_list.append(dist)
#    return torch.tensor(vocab_list, dtype=torch.long), torch.tensor(dist_list, dtype=torch.float)
    return torch.tensor(vocab_list).long(), torch.tensor(dist_list).float()
#    return vocab_list

def data_preprocess(smiles):
    property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - property_mean) / property_std
        atom_pair, dist = get_dist(smiles)
#        if any(x is None for x in (smiles, properties, atom_pair, dist)):
#            return None
        return (properties, '[CLS]' + smiles, atom_pair, dist)
    except Exception as e:
#        print(f"Failed processing {smiles}: {e}")
        return None

def safe_get_dist(args):
    property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
    idx, smi = args
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False, canonical=True)
        prop = ( calculate_property(smi) - property_mean) / property_std
        dist = get_dist(smi)
        return smiles, prop, dist
    except Exception as e:
        print(f'Failed {idx}: {e}')
        return None

if __name__ == '__main__':
    import time
    list_smi = [line.strip() for line in open('./data/1_Pretrain/1k_pretrain.txt')]
    st = time.time()
    with Pool(24) as p:
        results = list(tqdm(p.imap(safe_get_dist, enumerate(list_smi)), total=len(list_smi)))
    print('results!!!', len(results))
    #list_tmp = [item for row in results for item in row if None not in row]
    list_tmp = [item for item in results if None not in item]
    print('filtering \n\n', list_tmp)
    et = time.time()
    print(f'{len(list_tmp)} time: {et-st}')
    exit(-1)
#    output = [get_dist(smi) for smi in list_smi]
    error_idx =0
    for idx, smi in enumerate(tqdm(list_smi) ):
        try:
            atom_symbol, dist = get_dist(smi)
            #print(atom_symbol, dist)
        except Exception as e:
            error_idx +=1
            print(f'Faild {idx}: {e}')
    print(error_idx)
    exit(-1)
    results= []
    error=0
    for idx, smi in enumerate(tqdm(list_smi)):
        try:
            print(f'{idx}/{len(list_smi)}')
            atom_pair, dist = get_dist(smi)
        except:
            error+=1
            continue
    print(results)
    print(error)
    vocab_dict = Counter(results)
    print(vocab_dict)
    results= set(results)
    print(results)
    exit(-1)

#        results = data_preprocess(smi)
#        prop, smi, atom_pair, dist = results[0], results[1], results[2], results[3]
#        if prop is None:
#            prop_idx+=1
#        if smi_idx is None:
#            smi_idx+=1
#        if atom_idx is None:
#            atom_idx+=1
#        if dist_idx is None:
#            dist_idx+=1
#    print(prop_idx, smi_idx, atom_idx, dist_idx)

#        if data_preprocess(smi) is not None:
#            idx+=1
#            print(f'{idx}/{len(list_smi)}')
#    print(idx)
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
    
