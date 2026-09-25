#!/usr/bin/env python3
import argparse, ast
from pathlib import Path
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def can(s):
    m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
    return (m,Chem.MolToSmiles(m)) if m else (None,None)
def fp(m): return AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--sources',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('dirs',nargs='+',type=Path); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=False)
 tr_raw=set(pd.read_csv(a.train,header=None,dtype=str).iloc[:,2].dropna()); tr_can={c for _,c in map(can,tr_raw) if c}; rows=[]
 for gd in a.dirs:
  model=gd.parent.name; d=pd.read_csv(gd/'all_runs_statistics.csv'); cache={}
  for _,r in d.iterrows():
   target=str(r.protein_name)
   if target not in cache:
    src=pd.read_csv(a.sources/f'source_{target}.csv',header=None,dtype=str).iloc[:,0].dropna(); uniq={}
    for s in src:
     m,c=can(s)
     if m and c not in tr_can: uniq.setdefault(c,m)
    cache[target]=[fp(m) for m in uniq.values()]
   vals=ast.literal_eval(str(r.valid_smiles)); valid=[]
   for s in vals:
    m,c=can(s)
    if m: valid.append((s,m,c))
   unique_raw=set(x[0] for x in valid); novel_raw=unique_raw-tr_raw
   sims=[]
   for _,m,_ in valid:
    if cache[target]: sims.append(max(DataStructs.BulkTanimotoSimilarity(fp(m),cache[target])))
   rows.append({'model':model,'protein_name':target,'run_idx':int(r.run_idx),'valid_num':len(valid),'unique_raw_num':len(unique_raw),'novelty_gxvaes_percent':100*len(novel_raw)/len(unique_raw) if unique_raw else 0,'max_tanimoto_gxvaes':max(sims) if sims else 0})
 out=pd.DataFrame(rows); out.to_csv(a.output_dir/'all_runs_gxvaes_protocol.csv',index=False)
 best=out.loc[out.groupby(['model','protein_name']).max_tanimoto_gxvaes.idxmax()].sort_values(['model','protein_name']); best.to_csv(a.output_dir/'best_of_10_max_tanimoto_gxvaes_protocol.csv',index=False)
 print(best.groupby('model')[['max_tanimoto_gxvaes','novelty_gxvaes_percent']].mean().round(4).to_string()); print('\nPer target'); print(best.pivot(index='protein_name',columns='model',values='max_tanimoto_gxvaes').round(4).to_string())
if __name__=='__main__': main()
