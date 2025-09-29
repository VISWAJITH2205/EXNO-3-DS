Name : VISWAJITH LALITHRAM R.V

Reg.No : 212224240187

---

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

---

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/Encoding Data.csv")
df   
```

<img width="462" height="538" alt="Screenshot 2025-09-29 110841" src="https://github.com/user-attachments/assets/a07cba0b-30c4-4512-b3d8-35a7cd9622e7" />


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="612" height="565" alt="Screenshot 2025-09-29 111012" src="https://github.com/user-attachments/assets/76f57ab1-19f7-4fe0-881a-fce2a8378175" />


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="450" height="516" alt="Screenshot 2025-09-29 111108" src="https://github.com/user-attachments/assets/e2230de6-a194-4e33-8c88-40a8dcea88b4" />


```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="442" height="563" alt="Screenshot 2025-09-29 111157" src="https://github.com/user-attachments/assets/b3f04b10-a39b-45ca-9b3d-651676b8b92b" />


```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```

<img width="580" height="609" alt="Screenshot 2025-09-29 111243" src="https://github.com/user-attachments/assets/be1b66dd-2328-47c3-948c-3724f7df1572" />


```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="837" height="496" alt="Screenshot 2025-09-29 111329" src="https://github.com/user-attachments/assets/28b3a90f-db1a-48d1-805d-fa3f45400f40" />


```
!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("/data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="1218" height="809" alt="Screenshot 2025-09-29 111542" src="https://github.com/user-attachments/assets/8c824483-e83a-49c4-bb5b-0496382f4118" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="584" height="500" alt="Screenshot 2025-09-29 111653" src="https://github.com/user-attachments/assets/fac40042-1c02-40fd-b6ff-177c1ee673c8" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="807" height="528" alt="Screenshot 2025-09-29 111947" src="https://github.com/user-attachments/assets/58627730-b8a6-4121-a5a5-6ed412e6d585" />

```
df.skew()
```

<img width="359" height="247" alt="Screenshot 2025-09-29 112057" src="https://github.com/user-attachments/assets/6420017a-af9e-4030-8ed0-7303abc73f04" />

```
np.log(df["Highly Positive Skew"])
```

<img width="269" height="449" alt="Screenshot 2025-09-29 112231" src="https://github.com/user-attachments/assets/67a5edd9-eb78-4d23-b0a3-ed03cfed3a0a" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="367" height="491" alt="Screenshot 2025-09-29 112333" src="https://github.com/user-attachments/assets/62b23de5-8f63-4164-b6f8-0d8002a1c150" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="284" height="492" alt="Screenshot 2025-09-29 112437" src="https://github.com/user-attachments/assets/f7c73d47-461b-49af-a0b2-145fb09faf13" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1033" height="469" alt="Screenshot 2025-09-29 112626" src="https://github.com/user-attachments/assets/cb984439-4e89-4db9-8c49-900582c37eef" />

```
df.skew()
```

<img width="308" height="277" alt="Screenshot 2025-09-29 112826" src="https://github.com/user-attachments/assets/36cfb8ed-919c-4113-a16f-90ea8f997b8b" />

```
transformed_data, parameters = stats.yeojohnson(df["Highly Negative Skew"])
df["Highly Negative Skew_yeojohnson"] = transformed_data
df.skew()
```

<img width="610" height="343" alt="Screenshot 2025-09-29 113358" src="https://github.com/user-attachments/assets/1c6606f1-d259-4d1e-87cd-e34bf8769614" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1200" height="430" alt="Screenshot 2025-09-29 113539" src="https://github.com/user-attachments/assets/f6fb1c78-eb02-47f2-a736-5520137d8c08" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="505" height="471" alt="Screenshot 2025-09-29 113637" src="https://github.com/user-attachments/assets/cf201e15-ce16-4d7d-ad3e-fb308873277f" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="491" height="419" alt="Screenshot 2025-09-29 113807" src="https://github.com/user-attachments/assets/71c169f0-0c15-4033-8973-4ada852c23f5" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="510" height="470" alt="Screenshot 2025-09-29 113920" src="https://github.com/user-attachments/assets/78063f50-84b8-4746-be3d-68352633023d" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="501" height="438" alt="Screenshot 2025-09-29 114103" src="https://github.com/user-attachments/assets/f6e916a8-f4bd-42ed-9034-77e00553d3f5" />

```
dt=pd.read_csv("titanic_dataset.csv")
dt

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

<img width="487" height="470" alt="Screenshot 2025-09-29 114358" src="https://github.com/user-attachments/assets/bbcb01d9-bcb8-4186-9a6c-04a811943ec6" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="488" height="425" alt="Screenshot 2025-09-29 114508" src="https://github.com/user-attachments/assets/1b3bd2ec-ee41-447e-b9da-7e84a4a40db0" />

---

# RESULT:

  Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully

       
