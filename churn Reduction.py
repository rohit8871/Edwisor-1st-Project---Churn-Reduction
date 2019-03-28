#!/usr/bin/env python
# coding: utf-8

# In[288]:


#Loan Library
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import get_dummies


# # 1) Data collection

# In[363]:


# set working directory
os.chdir("D:/MY FIRST EDWISOR PROJECT\My first edwisor projects in python")


# In[364]:


# Load data
churn_train=pd.read_csv("Train_data.csv")

churn_test=pd.read_csv("Test_data.csv")


# In[365]:


churn_test.head(1)


# In[366]:


# print no. of customer in the data sets
print(" # of customers:" +str(len(churn_train.index)))


# # 2) Analyzing data 

# In[367]:


import seaborn as sns
# using count plot
sns.countplot(x="Churn",data=churn_train)


# In[368]:


sns.countplot(x="Churn",hue="international plan", data=churn_train)


# In[151]:


sns.countplot(x="Churn",hue="voice mail plan", data=churn_train)


# In[152]:


# Histogram
churn_train["account length"].plot.hist(bins=500,figsize=(10,5))


# In[153]:


churn_train["total day minutes"].plot.hist(bins=500,figsize=(10,5))


# In[154]:


churn_train.info()


# In[155]:


churn_train["number vmail messages"].plot.hist(bins=50,figsize=(10,5))


# In[156]:


churn_train["total day calls"].plot.hist(bins=500,figsize=(10,5))


# In[157]:


churn_train["total day minutes"].plot.hist(bins=50,figsize=(10,5))


# In[158]:


churn_train["total day charge"].plot.hist(bins=400,figsize=(10,5))


# In[159]:


churn_train["total eve minutes"].plot.hist(bins=300,figsize=(10,5))


# In[160]:


churn_train["total eve calls"].plot.hist(bins=500,figsize=(10,5))


# In[161]:


churn_train["total eve charge"].plot.hist(bins=500,figsize=(10,5))


# In[162]:


churn_train["total night minutes"].plot.hist(bins=500,figsize=(10,5))


# In[163]:


churn_train["total night calls"].plot.hist(bins=500,figsize=(10,5))


# In[164]:


churn_train["total night charge"].plot.hist(bins=500,figsize=(10,5))


# In[165]:


churn_train["total intl minutes"].plot.hist(bins=500,figsize=(10,5))


# In[166]:


churn_train["total intl charge"].plot.hist(bins=500,figsize=(10,5))


# In[167]:


churn_train["total intl calls"].plot.hist(bins=50,figsize=(10,5))


# In[168]:


churn_train["number customer service calls"].plot.hist(bins=100,figsize=(10,5))


# In[295]:


churn_train.columns


# In[170]:


churn_train.describe()


# # 3) Data wrangling

# In[369]:


#### Transform ctegorical data into factor in Train and Test data
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()


# In[370]:


#state variable 
churn_train["state"]=number.fit_transform(churn_train["state"].astype("str"))
churn_train.head(2)


# In[371]:


# state variable
churn_test["state"]=number.fit_transform(churn_test["state"].astype("str"))
churn_test.head(2)


# In[372]:


#international plan
churn_train["international plan"]=number.fit_transform(churn_train["international plan"].astype("str"))
churn_train.head(1)


# In[373]:


# international plan
churn_test["international plan"]=number.fit_transform(churn_test["international plan"].astype("str"))
churn_test.head(1)


# In[374]:


# voice mail plan
churn_train["voice mail plan"]=number.fit_transform(churn_train["voice mail plan"].astype("str"))
churn_train.head(1)

#voice mail plan
churn_test["voice mail plan"]=number.fit_transform(churn_test["voice mail plan"].astype("str"))
churn_test.head(1)
# In[375]:


# churn - train data
churn_train["Churn"]=number.fit_transform(churn_train["Churn"].astype("str"))
churn_train.head(1)


# In[376]:


# churn Test data
churn_test["Churn"]=number.fit_transform(churn_test["Churn"].astype("str"))
churn_test.head(1)


# # 3.1) Missing value Analysis

# In[360]:



# create dataframe with missing percentage
missing_churn_train=pd.DataFrame(churn_train.isnull().sum())
missing_churn_test=pd.DataFrame(churn_train.isnull().sum())


# In[305]:


#missing_churn_train  #   NO Any Missing Value Found
#missing_churn_test


# In[306]:


# Heat map to check missing 
sns.heatmap(churn_train.isnull(),yticklabels=False) # No missing value found


# # 3.2) Outlier Analysis

# In[307]:


churn_train.columns


# In[308]:


churn_train.describe()


# In[309]:


# plot boxplot to visualize outliers
  
sns.boxplot(x="Churn",y="total intl charge",data=churn_train)


# In[182]:


sns.boxplot(x="Churn",y="account length",data=churn_train)


# In[183]:


sns.boxplot(x="Churn",y="area code",data=churn_train)


# In[184]:


sns.boxplot(x="Churn",y="number vmail messages",data=churn_train)


# In[185]:


sns.boxplot(x="Churn",y="total day minutes",data=churn_train)


# In[186]:


sns.boxplot(x="Churn",y="total day calls",data=churn_train)


# In[187]:


sns.boxplot(x="Churn",y="total day charge",data=churn_train)


# In[188]:


sns.boxplot(x="Churn",y="total eve minutes",data=churn_train)


# In[189]:


sns.boxplot(x="Churn",y="total eve calls",data=churn_train)


# In[190]:


sns.boxplot(x="Churn",y="total eve charge",data=churn_train)


# In[191]:


sns.boxplot(x="Churn",y="total night minutes",data=churn_train)


# In[192]:


sns.boxplot(x="Churn",y="total night calls",data=churn_train)


# In[193]:


sns.boxplot(x="Churn",y="total night charge",data=churn_train)


# In[194]:


sns.boxplot(x="Churn",y="total intl minutes",data=churn_train)


# In[195]:


sns.boxplot(x="Churn",y="total intl calls",data=churn_train)


# In[196]:


sns.boxplot(x="Churn",y="total intl charge",data=churn_train)


# In[197]:


sns.boxplot(x="Churn",y="number customer service calls",data=churn_train)


# In[315]:


# save numeric names
#colnames=["account length","area code","number vmail messages","total day minutes","total day calls",
#          "total day charge","total eve minutes","total eve calls","total eve charge",
#          "total night minutes","total night calls","total night charge","total intl minutes",
#          "total intl calls","total intl charge","number customer service calls"]


# In[377]:


# Detect and delete outliers from data
#for i in colnames:
#   print(i)
#    q75,q25=np.percentile(churn_train.loc[:,i],[75,25]
#    iqr=q75-q25        #(iqr-> Inter Qualtile range : helps us to calculate the inner fence)
#    min=q25-(iqr*1.5)  # Lower fence
#    max=q75+(iqr*1.5)  # Upper fence
#    print(min)
#    print(max)
#    churn_train=churn_train.drop(churn_train[churn_train.loc[:,i]<min].index)
#    churn_train=churn_train.drop(churn_train[churn_train.loc[:,i]>max].index)


# In[378]:


#churn_train.describe()


# In[379]:


#churn_train.head(1)


# # 3.3) Feature selection

# In[380]:


#Load Library
import scipy.stats 
from scipy.stats import chi2_contingency


# In[381]:


## correlation analysis
# correlation plot
df_corr=churn_train.loc[:,colnames]

# set yhe width and height of plot
f,ax=plt.subplots(figsize=(20,6))
# Generate correlation Matrix
corr=df_corr.corr()
#plot using seaborn Library
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(250,10,as_cmap=True),
           square=True,ax=ax)


# In[382]:


## Chisquare test of independence
# save categorical variable
cat_names=['state','phone number','international plan', 'voice mail plan']


# In[383]:


churn_train.columns


# In[384]:


# Loop for chi_square values
for i in cat_names:
    print(i)
    chi,p,dof,ex=chi2_contingency(pd.crosstab(churn_train["Churn"],churn_train[i]))
    print(p)


# In[385]:


# delete variable carriying irrelevant information in Training data and test data
churn_train_deleted=churn_train.drop([ 'phone number','total day charge','total eve charge',
                                 'total night charge','total intl charge'],axis=1)

churn_test_deleted=churn_test.drop([ 'phone number','total day charge','total eve charge',
                                 'total night charge','total intl charge'],axis=1)


# In[386]:


churn_test_deleted.shape


# In[387]:


churn_train_deleted.head(2)


# # Apply Machine Learning Algorithm
# 

# In[ ]:





# In[415]:


churn_train_deleted.head(2)


# In[416]:


#### seperate the target variable in Training data
xTrain=churn_train_deleted.values[:,0:15]
xTrain


# In[419]:


xTrain.shape


# In[420]:


yTrain=churn_train_deleted.values[:,15]
yTrain


# In[421]:


#### seperate the target variable in test data
xTest=churn_test_deleted.values[:,0:15]
xTest


# In[422]:


yTest=churn_test_deleted.values[:,15]
yTest


# In[394]:


churn_train_deleted.head(2)


# # Decision Tree

# In[395]:


## Import libraries for decision tree
#from sklearn import tree
#from sklearn.metrics import accuracy_score


# In[400]:


#Decision Tree classifier
#clf=tree.DecisionTreeClassifier(criterion="entropy").fit(xTrain,yTrain)


# In[406]:


# predict new test cases
#y_predict=clf.predict(xTest)


# In[405]:


#xTest.head()


# In[135]:


#churn_train_deleted.head(5)


# #    Error  Metrix   of Decision Tree

# In[137]:


#Build confussion metrix
#from sklearn.metrics import confusion_matrix


# In[138]:


#CM_DT=pd.crosstab(yTest,y_predict)
#CM_DT
### by removing columns Total (day/eve/night/ intl) charge with outliers amd without scaling
#       False   True.
#False.	1376    67
#True.	69     155

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#col_0	False.	True.
#row_0		
#False.	1361	82
#True.	107	117


# In[139]:


# Let us save TP.TN,FP,FN
#TN=CM_DT.iloc[0,0]
#FN=CM_DT.iloc[1,0]
#TP=CM_DT.iloc[1,1]
#FP=CM_DT.iloc[0,1]


# In[140]:


# Check accuracy of model
#accuracy_score(yTest,y_predict)*100
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#91.84

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
88.66


# In[141]:


# False Negative Rate
(FN*100)/(FN+TP)
## by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#30.80

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#47.76


# In[142]:


# False positive rate
#(FP*100)/(FN+TN)
## by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#4.63

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#5.58


# In[143]:


#Recall
#(TP*100)/(TP+FN)


# # 2) Random Forest

# In[423]:


from sklearn.ensemble import RandomForestClassifier


# In[424]:


RF_model=RandomForestClassifier(n_estimators=100).fit(xTrain,yTrain)


# In[425]:


# prediction
RF_predictions=RF_model.predict(xTest)
RF_predictions


# # Error Matrix of Random Forest

# In[426]:


# Build confusion metrix
from sklearn.metrics import confusion_matrix
CM_RF=confusion_matrix(yTest,RF_predictions)
CM_RF


# In[119]:


CM_RF=pd.crosstab(yTest,RF_predictions)
CM_RF
## by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#col_0	False.	True.
#row_0
#False.	1437     6
#True.	74     150

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#col_0	False.	True.
#row_0		
#False.	1438	5
#True.	113	111


# In[120]:


# let us save TP,TN,FN,FP
TN=CM_RF.iloc[0,0]
FN=CM_RF.iloc[1,0]
TP=CM_RF.iloc[1,1]
FP=CM_RF.iloc[0,1]


# In[130]:


FN


# In[121]:


# check accuracy of model
# acauray_score(Y_test,Y_test)*100
((TP+TN)*100)/(TP+TN+FP+FN)
## by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#95.20

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#92.92


# In[122]:


# False Negative rate
(FN*100)/(FN+TP)
## by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#33.03

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
50.44


# In[123]:


# False positive rate
(FP*100)/(FN+TN)
## by removing columns charge (day,eve,night and intl) with outliers amd without scaling
# 0.39

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#0.32


# # 3) Logistic Regression

# In[428]:


# create Logistic data save target variables first in Training Data
churn_train_logit=pd.DataFrame(churn_train_deleted['Churn'])
churn_train_logit.head(2)


# In[429]:


# create Logistic data save target variables first in Test Data
churn_test_logit=pd.DataFrame(churn_test_deleted['Churn'])
churn_test_logit.head(2)


# In[430]:


churn_train.describe()


# In[431]:


# Continous Variable
cnames=["account length","area code",
        "number vmail messages","total day minutes","total day calls",
        "total eve minutes","total eve calls","total night minutes",
        "total night calls","total intl minutes","total intl calls",
        "number customer service calls"]


# In[432]:


# add continuous variable
churn_train_logit=churn_train_logit.join(churn_train_deleted[cnames])
churn_test_logit=churn_test_logit.join(churn_train_deleted[cnames])


# In[433]:


churn_test_logit.head(2)


# In[434]:


churn_train.head(1)


# In[435]:


# create  dummies for categorical variable in Training data
cat_names=["state","international plan","voice mail plan"]
for i in cat_names:
    temp=pd.get_dummies(churn_train[i],prefix=i)
    churn_train_logit=churn_train_logit.join(temp)


# In[436]:


# create  dummies for categorical variable in Test data
cat_names=["state","international plan","voice mail plan"]
for i in cat_names:
    temp=pd.get_dummies(churn_test[i],prefix=i)
    churn_test_logit=churn_test_logit.join(temp)


# In[437]:


churn_train_logit.head(2)


# In[438]:


churn_test_logit.head(2)


# In[439]:


#select column index for independent variables
train_cols=churn_train_logit.columns[1:68]
train_cols


# In[440]:


churn_train_logit.head(3)


# In[441]:


churn_train_logit[train_cols].head(2)


# In[442]:


# Built logistic regression
import statsmodels.api as sm


# In[443]:


logit=sm.Logit(churn_train_logit['Churn'], churn_train_logit[train_cols]).fit()


# In[444]:


#logit.summary()


# In[445]:


# predict test data
churn_test_logit['Actual_prob'] =logit.predict(churn_test_logit[train_cols])


# In[109]:


churn_test_logit.head(2)


# In[110]:


# convert praobability into target class
churn_test_logit['Actual_val']=1
churn_test_logit.loc[churn_test_logit.Actual_prob< 0.5 ,"Actual_val"]=0


# # Error Matrix of Logistic Regression

# In[112]:


# Built confusion matrix
CM_LR=pd.crosstab(churn_test_logit["Churn"],churn_test_logit["Actual_val"])
CM_LR
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#Actual_val 0       1
#Churn
#0	     1345      98
#1	     190       34

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#Actual_val	0	1
#Churn		
#0	1103	340
#1	166	58


# In[113]:


#Let us save TP,TN,FP,FN
TN=CM_LR.iloc[0,0]
FM=CM_LR.iloc[1,0]
TP=CM_LR.iloc[1,1]
FP=CM_LR.iloc[0,1]


# In[114]:


# check accuracy of model
#Accuracy_score(Y_test,Ypred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
(FN*100)/(FN+TP)
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#68.51

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
72.11


# # 4) KNN
# 

# In[270]:


#  Knn Implementation
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors=5).fit(xTrain,yTrain)


# In[271]:


# Predict test cases
knn_predictions=knn_model.predict(xTest)


# # Error Matrix of KNN

# In[272]:


# Build cinfusion matrix
CM_knn=pd.crosstab(yTest,knn_predictions)
CM_knn
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#col_0	False.	True.
#row_0
#False.	1290	153
#True.	142    82

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#col_0	False.	True.
#row_0		
#False.	1334   109
#True.	150    74


# In[265]:


#  Let us save TP,TN,FP,FN
TN=CM_knn.iloc[0,0]
FN=CM_knn.iloc[1,0]
TP=CM_knn.iloc[1,1]
FP=CM_knn.iloc[0,1]


# In[266]:


# check accuracy of model
accuracy_score(yTest,knn_predictions)*100
((TP+TN)*100)/(TP+TN+FN+FP)
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
# 85.80

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#84.46


# In[88]:


# false negative rate
(FN*100)/(FN+TP)
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
# 47.43

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#66.96


# # 5) Naive Bayes

# In[73]:


#NaiveBays
from sklearn.naive_bayes import GaussianNB


# In[74]:


# NAive bayes  implementation
NB_model=GaussianNB().fit(xTrain,yTrain)


# In[75]:


# Predict  test   cases 
NB_predictions=NB_model.predict(xTest)


# # Error Matrix of Naive Bayes

# In[76]:


# Build confusion metrix
CM_NB=pd.crosstab(yTest,NB_predictions)
CM_NB
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
#col_0	False.	True.
#row_0		
#False.	1342	101
#True.	135     89

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#col_0	False.	True.
#row_0		
#False.	1366    77
#True.	147    77


# In[79]:


#  Let us save TP,TN,FP,FN
TN=CM_NB.iloc[0,0]
FN=CM_NB.iloc[1,0]
TP=CM_NB.iloc[1,1]
FP=CM_NB.iloc[0,1]


# In[80]:


# check accuracy  of model
# accuracy_score (Y_test,Y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
# 89.10

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#86.56


# In[81]:


# false negative rate
(FN*100)/(FN+TP)
#### by removing columns charge (day,eve,night and intl) with outliers amd without scaling
# FNR=45.39

###  removing columns Total (day/eve/night/intl) charge removing outliers and without scaling
#65.62


# In[ ]:




