import pandas as pd
from chembl_webresource_client.new_client import new_client
import json
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import numpy as np
from rdkit.DataStructs import ConvertToNumpyArray
import streamlit as st 

#Get target proteins for Influenza A virus 
target = new_client.target
viral_targets = target.filter(target_type='SINGLE PROTEIN', organism='Influenza A Virus')
targets = pd.DataFrame(list(viral_targets))
viral_targets_ids=[t['target_chembl_id'] for t in viral_targets]

# Clean data, conserving only relevant targets such as conficence values above seven and stage three, meaning that they are tested compounds 
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
df=pd.DataFrame(all_viral_activities)
df.to_csv('all_viral_activities.csv', index=False)

#Dropping null values 
df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]
#Dropping duplicates based on canonical smiles 
len(df2.canonical_smiles.unique())
df2_nr = df2.drop_duplicates(['canonical_smiles'])

#Relevanr columns for the analysis 
selection= ['pref_name','molecule_chembl_id', 'canonical_smiles','standard_value']
df3=df2_nr[selection]
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

#Creating individual lists for each column, the adding the activity threshold to a new dataframe. 
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

df5.to_csv(r'processed_data.csv', index=False)

#Sorting the dataset based on activity 
df6= pd.read_csv("processed_data.csv")
bioactivity_order=['active','intermediate','inactive'] #rearranging the organisms from active to inactive 
df6['bioactivity_threshold']=pd.Categorical(df6['bioactivity_threshold'], categories=bioactivity_order,ordered=True)
df_sorted_data=df6.sort_values(by='bioactivity_threshold')
print(df_sorted_data)
df_sorted_data.to_csv('sorted_data.csv', index=False)


IC50_values=df_sorted_data['standard_value']
ic50_M=IC50_values/1e9
df_sorted_data['PIC50']= -np.log10(ic50_M)
print("IC50 + PIC50 Values")
print(df_sorted_data)
df_sorted_data.to_csv('sorted_data.csv', index=False)


# #FIltering to only get active compunds with a standard value below 50.0
active_compunds=df_sorted_data[df_sorted_data['bioactivity_threshold']=='active']
filtered_compounds=active_compunds[active_compunds['standard_value']<50.0]
active_filtered = filtered_compounds.sort_values(by='standard_value', ascending=False)
print("Filtered + Sorted Compounds")
print(active_filtered)
active_filtered.to_csv('active_filtered_compounds.csv', index=False)

# #Get the fingerprints from the molecules
compounds=pd.read_csv("active_filtered_compounds.csv")
compounds['mol'] = compounds['canonical_smiles'].apply(lambda smiles: Chem.MolFromSmiles(smiles))  # Convert SMILES to RDKit molecules
compounds['ECFP4']=compounds['canonical_smiles'].apply(lambda smiles:  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=2048).ToBitString()) 
fingerprints = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)) for mol in compounds['mol']], dtype=int)
np.save("fingerprints.npy",fingerprints)
#Get Chemical Descriptors 
compounds['MolWt'] = compounds['mol'].apply(Descriptors.MolWt)
compounds['LogP'] = compounds['mol'].apply(Descriptors.MolLogP)
compounds['NumHDonors'] = compounds['mol'].apply(Descriptors.NumHDonors)
compounds['NumHAcceptors'] = compounds['mol'].apply(Descriptors.NumHAcceptors)
compounds['TPSA'] = compounds['mol'].apply(Descriptors.TPSA)
compounds['NumRotatableBonds'] = compounds['mol'].apply(Descriptors.NumRotatableBonds)

compounds.to_csv(r'Data_fingerprints.csv', index=False)

df_fingerprints=pd.read_csv("Data_fingerprints.csv")
df_fingerprints['ECFP4'] = df_fingerprints['ECFP4'].dropna().apply(lambda x: DataStructs.CreateFromBitString(str(x)))
df_fingerprints.to_csv('data_to_display.csv', index=False)
# #Generating images + Label 
df_data_to_display=pd.read_csv("data_to_display.csv")
df_data_to_display['mol']=df_data_to_display['canonical_smiles'].apply(Chem.MolFromSmiles)
df_data_to_display[df_data_to_display['mol'].notna()]
df_data_to_display['legend'] = df_data_to_display.apply(lambda row: f"{str(row['pref_name'])}\n{str(row['bioactivity_threshold']).capitalize()} | IC50:{row['standard_value']} µM", axis=1)


st.title("Antiviral Compound Only Active Molecules")
num_to_display=st.slider("Number of Molecules to display", 5,30,10)

img=Draw.MolsToGridImage(df_data_to_display['mol'].head(num_to_display).tolist(),subImgSize=(300,300), legends=df_data_to_display['legend'].head(num_to_display).tolist())
st.image(img)
st.title("Active Molecules")
st.dataframe(df_data_to_display)
