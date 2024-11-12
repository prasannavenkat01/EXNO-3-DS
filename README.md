## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed  By: PRASANNA V 
Register Number : 212223240123
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/4d435068-e78f-4d3c-83dc-ae9b7630a967)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/03942a61-d367-453c-8eb1-4fa2d927b01b)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/2701967c-2909-49af-afd7-04057cb75573)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/13579943-94e1-4a79-906a-c16ec097ad9a)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/user-attachments/assets/c5a6c2cc-bf78-4c19-89a5-1d25386a6bba)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/41047d3e-fe86-43f4-9c6c-7e86dec05b2a)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/64fd2af5-3b73-40ae-b977-51eb11d242f9)

```
pip install --upgrade category_encoders
```
 ![image](https://github.com/user-attachments/assets/accc99ee-2a01-4602-8e9d-bd2858d604ad)

 ```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/62c423ed-af97-4a64-b1d9-41464769c37b)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/7b49e579-68ce-4146-a9e2-4a1dc7bcf548)


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/cd06e136-b204-4a49-a132-254583ae35d4)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/00c687a6-61a9-4ae0-83b1-470c3a163eed)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/1cd6ccfe-c909-4278-9891-e96eb04b2027)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f9b0c4fd-850e-4514-9426-9bdc416647d9)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1983a683-a975-461a-958b-7d1676cfbcf5)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/37b7db91-16f7-4ba9-beb1-268a621f221e)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/6b3377fb-d296-4071-b0d6-9246f56f4f05)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/user-attachments/assets/88cc6f76-5305-4058-ab8f-5064849dea36)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/dd3fe9bc-b64c-4d96-9542-78bf19c0dbe3)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b4769efe-98eb-47a6-86f4-20320a789120)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/11580c9d-9684-4320-9621-1f885a0a4ba6)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9938d969-22ca-4d93-8f74-d1aaba64db60)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/61f75d73-ec15-43fe-8bab-4466b6c2575c)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/17f05d83-eb96-4c73-aa69-c4ac6a602f08)

# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.



























       
