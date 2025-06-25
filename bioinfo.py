import pandas as pd
import datetime as dt 
from chembl_webresource_client.new_client import new_client# Target search for coronavirus
import json
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import AllChem
from rdkit import Chem
import pubchempy as pcp
import sqlite3
#from rdkit.Chem import MorganGenerator

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the Morgan fingerprint generator
from rdkit.Chem import Draw
import numpy as np
from rdkit.DataStructs import ConvertToNumpyArray


target = new_client.target
viral_targets = target.filter(target_type='SINGLE PROTEIN').filter(organism__icontains='virus')
targets = pd.DataFrame(list(viral_targets))
print(targets[['target_chembl_id','pref_name','target_type']])

molecule = new_client.molecule
approved_molecules=molecule.filter(max_phase=4)
approved_ids=set(m['molecule_chembl_id'] for m in approved_molecules)

selected_target=targets.target_chembl_id.iloc[0]


activity=new_client.activity
all_viral_activities=[]
for target_id in viral_target_ids:
    res = activity.filter(target_chembl_id=target_id, assay_type="B",standard_type="'IC50", confidence_score_gte=7).only(
    ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units', 'activity_comment'])

    for r in res:
        if r.get('molecule_chembl_id') in approved_ids and r.get('standard_relation') in [None, '=']:
            try:
                r['standard_value'] = float(r['standard_value'])
                all_viral_activities.append(r)
            except:
                continue

df = pd.DataFrame(all_viral_activities)

# Preview structure
print(df.columns)
print(df.head())

df.to_csv('bioactivity_data.csv', index=False )


df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]
df2
len(df2.canonical_smiles.unique())
df2_nr = df2.drop_duplicates(['canonical_smiles'])
df2_nr
selection= ['molecule_chembl_id', 'canonical_smiles','standard_value']
df3=df2_nr[selection]
df3.to_csv('bioactivity_data_preprocessed.csv', index=False )


df4=pd.read_csv('bioactivity_data_preprocessed.csv')


bioactivity_threshold = []

for i in df4.standard_value:
  if float(i) >= 10000:
    bioactivity_threshold.append("inactive")
  elif float(i) <= 1000:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")
mol_cid = []
for i in df2.molecule_chembl_id:
  mol_cid.append(i)

mol_cid=[]
for i in df2.molecule_chembl_id:
  mol_cid.append(i)
canonical_smiles=[]
for i in df4.canonical_smiles:
  canonical_smiles.append(i)

standard_value=[]
for i in df4.standard_value:
  standard_value.append(i)
  

data_tuples=list(zip(mol_cid,canonical_smiles,bioactivity_threshold,standard_value))
df5=pd.DataFrame(data_tuples,columns=['molecule_chembl_id','canonical_smiles','bioactivity_threshold','standard_value'])
#print(bioactivity_threshold)
df5
df5.to_csv(r'processed_data.csv', index=False)
print(df5)

#From now on the processes must be done based on the clean dataset
df= pd.read_csv("processed_data.csv")
df_data=pd.DataFrame(df)
bioactivity_order=['active','intermediate','inactive'] #rearranging the organisms from active to inactive 
df['bioactivity_threshold']=pd.Categorical(df['bioactivity_threshold'], categories=bioactivity_order,ordered=True)
df_sorted=df.sort_values(by='bioactivity_threshold')
print("Sorted Dataset")
print(df_sorted)

#Converting the IC50 data to PIC50 for standarization

IC50_values=df_sorted['standard_value']
print("Values")
print(IC50_values)
ic50_M=IC50_values/1e9
df['PIC50']= -np.log10(ic50_M)
print("IC50 + PIC50 Values")
print(df)


#Getting Fingerprints from SMILES 
df['mol'] = df['canonical_smiles'].apply(lambda smiles: Chem.MolFromSmiles(smiles))  # Convert SMILES to RDKit molecules
df['ECFP4']=df['canonical_smiles'].apply(lambda smiles:  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=2048).ToBitString()) 
fingerprints = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)) for mol in df['mol']], dtype=int)
np.save("fingerprints.npy",fingerprints)
df.to_csv(r'processed_data1.csv', index=False)
#print(type(df['ECFP4'].iloc[0]))
df=pd.read_csv("processed_data1.csv")
df['ECFP4'] = df['ECFP4'].dropna().apply(lambda x: DataStructs.CreateFromBitString(str(x)))
#print(type(df['ECFP4'].iloc[0])) 
#print(df[['ECFP4']].head())

#Labeling Rdkit images 
#smiles=df['canonical_smiles']
#compounds=pcp.get_compounds(smiles,namespace='smiles')
#if compounds:
 # print("Name: ",compounds[0].iupac_name or compounds[0].synonyms[0])





#Model Training 
#Defining variables x and y 
#x = np.vstack(df['ECFP4'].tolist())
#y=df['PIC50']
#print(x.shape, y.shape)


#Testing variables 
#x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2,random_state=42)
#print("training set: ", x_train.shape, "Test set: ", x_test.shape)


#Initializing model to test 
#model=RandomForestRegressor(n_estimators=100, random_state=42)
#model.fit(x_train, y_train)
#y_pred=model.predict(x_test)



#Evaluating Performance: 
#print("MAE:", mean_absolute_error(y_test, y_pred))
#print("MSE:", mean_squared_error(y_test, y_pred))
#print("R² Score:", r2_score(y_test, y_pred))



#Getting Image 
df['mol']=df['canonical_smiles'].apply(Chem.MolFromSmiles)
df['legend'] = df.apply(lambda row: f"{row['molecule_chembl_id']}\n{row['bioactivity_threshold'].capitalize()}| IC50:{row['standard_value']} µM", axis=1)
img=Draw.MolsToGridImage(df['mol'].head(13).tolist(),molsPerRow=13,subImgSize=(300,300), legends=df['legend'].head(13).tolist())
img.show()
img
img.save("output.png")

