import drug_search
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import DataStructs
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns



df= pd.read_csv("sorted_data.csv")
df["IC50_M"]=df['standard_value']/1e9
df["IC50_M"] = df["IC50_M"].where(df["IC50_M"] > 0, np.nan)

df["PIC50"] =  -np.log10(df["IC50_M"])
df = df.dropna(subset=["PIC50"])
df["PIC50"]= df["PIC50"].astype("float64")

df.to_csv('data_to_read.csv', index=False)
df=pd.read_csv("data_to_read.csv")
#Model Training 
#Defining variables x and y 
df = df[df['ECFP4'].notnull()]
df['ECFP4'] = df['ECFP4'].dropna().apply(lambda x: DataStructs.CreateFromBitString(str(x)))

x = np.vstack(df['ECFP4'].tolist())
y=df['PIC50']
print(x.shape, y.shape)


#Testing variables 
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2,random_state=42)
print("training set: ", x_train.shape, "Test set: ", x_test.shape)


#Initializing model to test 
model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)



#Evaluating Performance: 
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


plt.figure(figsize=(7,7))
sns.scatterplot(x=y_test,y=y_pred)
plt.plot([y.min(),y.max()],[y.min(),y.max()], 'r--')
plt.xlabel("Actual pIC₅₀(test set)")
plt.ylabel("Predicted pIC₅₀(test set)")
plt.title("Predicted vs. Actual pIC₅₀ on test set")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=300,bbox_inches='tight')

residuals = y - model.predict(x)
plt.figure(figsize=(7, 4))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.grid(True)
plt.savefig("pic50_plot.png", dpi=300,bbox_inches='tight')
plt.close()


st.title("Prediction Model with RandomForest")
st.image("pic50_plot.png", caption="The Model Calibration Plot")
st.image("predicted_vs_actual.png", caption="Predicted vs. Actual pIC₅₀")


