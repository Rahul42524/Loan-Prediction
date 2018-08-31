# Importing the liberaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset using pandas
train_data = pd.read_csv('train.csv')  

train_data.shape # And shape of the Train_data is rows=614 and column=13 

train_data.head(5) # 'head' and 'tail' function give the tov record and bottom record respectively.

train_data.describe() # describe() shows a quick statistic summary of the data
"""
       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  .......
count       614.000000         614.000000  592.000000         600.00000  .......
mean       5403.459283        1621.245798  146.412162         342.00000  .......
std        6109.041673        2926.248369   85.587325          65.12041  ....... 
min         150.000000           0.000000    9.000000          12.00000  ....... 
25%        2877.500000           0.000000  100.000000         360.00000  ....... 
50%        3812.500000        1188.500000  128.000000         360.00000  ....... 
75%        5795.000000        2297.250000  168.000000         360.00000  ....... 
max       81000.000000       41667.000000  700.000000         480.00000  ....... """


train_data.columns # columns function display the colums
"""Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],"""

train_data.apply(lambda x: sum(x.isnull()),axis=0)  # Checking missing values in data 
#Note: in LoanAmount column 0 is also missing value bacause loan amount never 0
"""Loan_ID            0
Gender               13
Married               3
Dependents           15
Education             0
Self_Employed        32
ApplicantIncome       0
CoapplicantIncome     0
LoanAmount           22
Loan_Amount_Term     14
Credit_History       50
Property_Area         0
Loan_Status           0
dtype: int64
"""



# Take caring of missing data
train_data['Gender'].fillna('No',inplace=True)
train_data['Married'].fillna('No',inplace=True)
train_data.Dependents.fillna(0,inplace=True)
train_data['Self_Employed'].fillna('No',inplace=True)
train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mean(), inplace=True)
train_data['Credit_History'].fillna('No',inplace=True)


train_data.apply(lambda x: sum(x.isnull()),axis=0)
""" 
Loan_ID              0
Gender               0
Married              0
Dependents           0
Education            0
Self_Employed        0
ApplicantIncome      0
CoapplicantIncome    0
LoanAmount           0
Loan_Amount_Term     0
Credit_History       0
Property_Area        0
Loan_Status          0
dtype: int64
"""


#converting sring value into numerical values

train_data.Gender.value_counts()
gender_cat = pd.get_dummies(train_data.Gender,prefix='gender').gender_Female

train_data.Married.value_counts()
married_category = pd.get_dummies(train_data.Married,prefix='marriage').marriage_Yes

train_data.Education.value_counts()
graduate_category = pd.get_dummies(train_data.Education,prefix='education').education_Graduate


train_data.Self_Employed.value_counts()
self_emp_category = pd.get_dummies(train_data.Self_Employed,prefix='employed').employed_Yes

loan_status = pd.get_dummies(train_data.Loan_Status,prefix='status').status_Y

property_category = pd.get_dummies(train_data.Property_Area,prefix='property')

train_data.shape #(614, 13)

train_data.head(5)
"""
 ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term   credit_History Property_Area Loan_status
0             5849                0.0  146.412162             360.0     1            Urban          Y
1             4583             1508.0  128.000000             360.0     1            Rural          N
2             3000                0.0   66.000000             360.0     1            Urban          Y
3             2583             2358.0  120.000000             360.0     1            Urban          Y
4             6000                0.0  141.000000             360.0     1            Urban          Y
"""

New_trainData = pd.concat([train_data,gender_cat,graduate_category,married_category,self_emp_category,loan_status,property_category], axis=1)

New_trainData.shape  #(614, 21)

New_trainData.columns
"""
(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status',
       'gender_Female', 'education_Graduate', 'marriage_Yes', 'employed_Yes',
       'status_Y', 'property_Rural', 'property_Semiurban', 'property_Urban'],
      dtype='object')
"""

feature_columns = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','gender_Female','marriage_Yes', 'education_Graduate','employed_Yes','property_Rural','property_Semiurban','property_Urban']

X = New_trainData[feature_columns]
X.shape #(614, 12)

y =  New_trainData['status_Y']
y.shape #(614,)

# Splitting into traning snd test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

X_train.shape  #(491, 11)
y_train.shape   #(491,)

X_test.shape   #(123, 11)
y_test.shape   #(123,)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

randForest = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForest.fit(X_train,y_train)
y_pred_class  = randForest.predict(X_test)
randForestScore = accuracy_score(y_test,y_pred_class)

print(randForestScore)
"""
Random forest accuraccy score 0.96918699186991873
"""

#-------------------------Importing test data----------------------------------- 

randForestNew = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForestNew.fit(X,y)

test_data=pd.read_csv('test.csv')

test_data.shape # (367, 12)

test_data.columns 
"""
(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'],
      dtype='object')
"""
test_data.head(5)

test_data.describe()
"""
      ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \
count       367.000000         367.000000  362.000000        361.000000   
mean       4805.599455        1569.577657  136.132597        342.537396   
std        4910.685399        2334.232099   61.366652         65.156643   
min           0.000000           0.000000   28.000000          6.000000   
25%        2864.000000           0.000000  100.250000        360.000000   
50%        3786.000000        1025.000000  125.000000        360.000000   
75%        5060.000000        2430.500000  158.000000        360.000000   
max       72529.000000       24000.000000  550.000000        480.000000   
"""

test_data.apply(lambda x: sum(x.isnull()),axis=0)

"""
Loan_ID               0
Gender               11
Married               0
Dependents           10
Education             0
Self_Employed        23
ApplicantIncome       0
CoapplicantIncome     0
LoanAmount            5
Loan_Amount_Term      6
Credit_History       29
Property_Area         0
dtype: int64
"""

# Take caring o missing data

test_data['Gender'].fillna('No',inplace=True)
test_data.Dependents.fillna(0,inplace=True)
test_data['Self_Employed'].fillna('No',inplace=True)
test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean(), inplace=True)
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mean(), inplace=True)
test_data['Credit_History'].fillna('No',inplace=True)

test_data.apply(lambda x: sum(x.isnull()),axis=0)
"""
Loan_ID              0
Gender               0
Married              0
Dependents           0
Education            0
Self_Employed        0
ApplicantIncome      0
CoapplicantIncome    0
LoanAmount           0
Loan_Amount_Term     0
Credit_History       0
Property_Area        0
dtype: int64
"""


#converting sring value into numerical values

test_data.Gender.value_counts()
gender_cat = pd.get_dummies(test_data.Gender,prefix='gender').gender_Female

test_data.Married.value_counts()
married_category = pd.get_dummies(test_data.Married,prefix='marriage').marriage_Yes

test_data.Education.value_counts()
graduate_category = pd.get_dummies(test_data.Education,prefix='education').education_Graduate


test_data.Self_Employed.value_counts()
self_emp_category = pd.get_dummies(test_data.Self_Employed,prefix='employed').employed_Yes

property_category = pd.get_dummies(test_data.Property_Area,prefix='property')

test_data.head(5) 

test_data.columns


New_testData = pd.concat([test_data,gender_cat,graduate_category,married_category,self_emp_category,property_category], axis=1)

New_testData.shape  #(614, 21)

New_trainData.columns
"""
(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status',
       'gender_Female', 'education_Graduate', 'marriage_Yes', 'employed_Yes',
       'status_Y', 'property_Rural', 'property_Semiurban', 'property_Urban'],
      dtype='object')
"""

# Creating test data according to feature columns
x_test = New_testData[feature_columns]
x_test.shape #(367, 12)
x_test.columns
"""
(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'gender_Female', 'marriage_Yes',
       'education_Graduate', 'employed_Yes', 'property_Rural',
       'property_Semiurban', 'property_Urban'],
      dtype='object')
"""

# predicting loan status of Applicants
y_test_pread_class = randForestNew.predict(x_test)
randForestFormat = ["Y" if i == 1 else "N" for i in y_test_pread_class ] # yes=1 and No=0
pd.DataFrame({'Loan_ID':test_data.Loan_ID,'Loan_Status':randForestFormat}).to_csv('radom_forest_submission.csv',index=False)

print(y_test_pread_class)




