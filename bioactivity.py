import pandas as pd
from chembl_webresource_client.new_client import new_client#This program takes in a database with a certain virus, in this case, coronavirus, and classifies its drugs, compounds and effectivity. Then stores the new database to a csv document as preprocessed data. 
import json
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
selected_target=targets.target_chembl_id.iloc[4]
selected_target
print(targets)


activity=new_client.activity
res=activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")#selects the ic50 values only 
#print(res)

df = pd.DataFrame.from_dict(res)
df.head(3)
df.standard_type.unique()
df.to_csv('bioactivity_data.csv', index=False)


df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]
df2
len(df2.canonical_smiles.unique())
df2_nr = df2.drop_duplicates(['canonical_smiles'])
df2_nr
# selection=['molecule_chembl_id','canonical_smiles','standard_value']
# df3 = df2_nr[selection]
# df3
# df3.to_csv(r'C:\Users\Sofia\Downloads\acetylcholinesterase_02_bioactivity_data_preprocessed.csv', index=False)
bioactivity_threshold = []

# df4 = pd.read_csv('acetylcholinesterase_02_bioactivity_data_preprocessed.csv')

for i in df2.standard_value:
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
for i in df2.canonical_smiles:
  canonical_smiles.append(i)

standard_value=[]
for i in df2.standard_value:
  standard_value.append(i)

data_tuples=list(zip(mol_cid,canonical_smiles,bioactivity_threshold,standard_value))
df3=pd.DataFrame(data_tuples,columns=['molecule_chembl_id','canonical_smiles','bioactivity_threshold','standard_value'])

df3
df3.to_csv(r'C:\Users\sofia\Downloads\bioactivity_preprocessed_data.csv', index=False)

