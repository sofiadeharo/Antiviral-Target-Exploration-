import pandas as pd
import datetime as dt 
from chembl_webresource_client.new_client import new_client# Target search for coronavirus
import json
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import AllChem
from rdkit import Chem
import pubchempy as pcp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the Morgan fingerprint generator
from rdkit.Chem import Draw
import numpy as np
from rdkit.DataStructs import ConvertToNumpyArray
import streamlit as st 

target = new_client.target
viral_targets = target.filter(target_type='SINGLE PROTEIN', organism='Influenza A virus')
targets = pd.DataFrame(list(viral_targets))
viral_targets_ids=[t['target_chembl_id'] for t in viral_targets]


activity=new_client.activity
all_viral_activities=[]
for target_id in viral_targets_ids:
    target_name=targets[targets['target_chembl_id']== target_id]['pref_name'].values[0]
    res = activity.filter(target_chembl_id=target_id, assay_type="B",standard_type="IC50", confidence_score_gte=7, stage=3).only(
    ['pref_name','molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units', 'activity_comment'])
    for r in res:
        # Only keep exact numeric values
        if r.get('standard_relation') in [None, '=']:
            try:
                r['standard_value'] = float(r['standard_value'])
                r['pref_name'] = target_name 
                all_viral_activities.append(r)
            except (ValueError, TypeError):
                continue

df = pd.DataFrame(all_viral_activities)



df.to_csv('data.csv', index=False )


df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]

len(df2.canonical_smiles.unique())
df2_nr = df2.drop_duplicates(['canonical_smiles'])


selection= ['pref_name','molecule_chembl_id', 'canonical_smiles','standard_value']
df3=df2_nr[selection]

print(df3)
df3.to_csv('bioactivity_data.csv', index=False )


df4=pd.read_csv('bioactivity_data.csv')


bioactivity_threshold = []

for i in df4.standard_value:
  if float(i) >= 10000:
    bioactivity_threshold.append("inactive")
  elif float(i) <= 1000:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")

mol_cid=[]
for i in df4.molecule_chembl_id:
  mol_cid.append(i)
canonical_smiles=[]
for i in df4.canonical_smiles:
  canonical_smiles.append(i)

standard_value=[]
for i in df4.standard_value:
  standard_value.append(i)
pref_name=[]
for i in df4.pref_name:
   pref_name.append(i)

data_tuples=list(zip(mol_cid,canonical_smiles,bioactivity_threshold,standard_value,pref_name))
df5=pd.DataFrame(data_tuples,columns=['molecule_chembl_id','canonical_smiles','bioactivity_threshold','standard_value','pref_name'])

df5.to_csv(r'processed_data_new.csv', index=False)
#From now on the processes must be done based on the clean dataset
df= pd.read_csv("processed_data_new.csv")
df_data=pd.DataFrame(df)
bioactivity_order=['active','intermediate','inactive'] #rearranging the organisms from active to inactive 
df['bioactivity_threshold']=pd.Categorical(df['bioactivity_threshold'], categories=bioactivity_order,ordered=True)
df_sorted=df.sort_values(by='bioactivity_threshold')
print("Sorted Dataset")
print(df_sorted)

active_df=df_sorted[df_sorted['bioactivity_threshold']=='active']
filtered_df=active_df[active_df['standard_value']<50.0]

df_sorted_data = filtered_df.sort_values(by='standard_value', ascending=False)
print("Filtered + Sorted Compounds")
print(df_sorted_data)

IC50_values=df_sorted_data['standard_value']
print("Values")
print(IC50_values)
ic50_M=IC50_values/1e9
df_sorted_data['PIC50']= -np.log10(ic50_M)
print("IC50 + PIC50 Values")
print(df_sorted_data)
df_sorted_data.to_csv('sorted_data.csv', index=False)


df['mol'] = df['canonical_smiles'].apply(lambda smiles: Chem.MolFromSmiles(smiles))  # Convert SMILES to RDKit molecules
df['ECFP4']=df['canonical_smiles'].apply(lambda smiles:  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=2048).ToBitString()) 
fingerprints = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)) for mol in df['mol']], dtype=int)
np.save("fingerprints.npy",fingerprints)

df.to_csv(r'sorted_data.csv', index=False)
df=pd.read_csv("sorted_data.csv")
df['ECFP4'] = df['ECFP4'].dropna().apply(lambda x: DataStructs.CreateFromBitString(str(x)))
df_sorted_data['PIC50']= -np.log10(ic50_M)



df = df_sorted_data.copy()
df['mol']=df['canonical_smiles'].apply(Chem.MolFromSmiles)
df = df[df['mol'].notna()]
df_sorted_data['legend'] = df_sorted_data.apply(lambda row: f"{str(row['pref_name'])}\n{str(row['bioactivity_threshold']).capitalize()} | IC50:{row['standard_value']} µM", axis=1)

st.title("Antiviral Compound Explorer")
num_to_display=st.slider("Number of Molecules to display", 5,30,10)
img=Draw.MolsToGridImage(df['mol'].head(num_to_display).tolist(),subImgSize=(300,300), legends=df_sorted_data['legend'].head(num_to_display).tolist())
st.image(img)
st.title("Data")
st.dataframe(df_sorted_data)


