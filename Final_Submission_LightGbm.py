##Team 34 Kaggle Group Project Income Prediction Python Code File##

#Modules used in the project
import pandas as pd
import lightgbm as lgb
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

#import Train and Test Data from Excel
train = pd.read_csv("tcd-ml-1920-group-income-train.csv",na_values=['Unknown','nA','#NA','#NUM!','unknown','#N/A'])
test = pd.read_csv("tcd-ml-1920-group-income-test.csv",na_values=['Unknown','nA','#NA','unknown','#NUM!','#N/A'])

                                                                  
#Data Cleaning
###Remove unnecessary columns and duplicates###
#Removing Crime column and Wear glasses columns after checking the correlation of these columns with Dependent Column Income
test = test.drop(['Wears Glasses','Crime Level in the City of Employement'],axis=1)
train = train.drop(['Wears Glasses','Crime Level in the City of Employement'],axis=1)
#Removing duplicate entries in train and test data to improve the accuracy of prediction by eliminating the duplicates
train = train.drop_duplicates()                                     
test = test.drop_duplicates() 

#Removing EUR keyword from 'Yearly Income in addition to Salary (e.g. Rental Income)' column to treat missing values n the column
test=test.replace(to_replace=' EUR', value='', regex=True)
test['Yearly Income in addition to Salary (e.g. Rental Income)'] = test['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
train=train.replace(to_replace=' EUR' , value='', regex=True) 
train['Yearly Income in addition to Salary (e.g. Rental Income)'] = train['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)


#List of columns which are not of type object
def constantsList(data):
    c=[]
    for col in data.columns:
        if data[col].dtypes!="object":
                c.append(col)
    return c

test_without_income = test.iloc[:,:-1]
train_without_income = train.iloc[:,:-1]

#Replacing missing values with mean
constantColumns=constantsList(train_without_income)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
for con in constantColumns:
    train_without_income[con]=imputer.fit_transform(train_without_income[con].values.reshape(-1,1))
    test_without_income[con]=imputer.fit_transform(test_without_income[con].values.reshape(-1,1))

#Replace 'f' with female in Gender and 0 with unknown
train["Gender"]= train["Gender"].replace('0','unknown') 
train["Gender"]= train["Gender"].replace('f','female')  
test["Gender"]= test["Gender"].replace('0','unknown') 
test["Gender"]= test["Gender"].replace('f','female')

#Treating Categorical Column NA values as missing
def fillCategoricalsWithMissing(data):
    fill_col_dict = {
        'Housing Situation': 'missing',
        'Satisfation with employer':'missing',
        'Gender':'missing',
        'Country':'missing',
        'Profession':'missing',
        'University Degree':'missing',
        'Hair Color':'missing',
    }
    for col in fill_col_dict.keys():
        data[col] = data[col].fillna(fill_col_dict[col])
    
fillCategoricalsWithMissing(train)
fillCategoricalsWithMissing(test)

#Encoding the data using Target Encoding Approach
target = TargetEncoder()
train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
target.fit(train_X,train_y)
train_X = target.transform(train_X)
test_without_income = target.transform(test_without_income)
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=4321)

#Training model with LightGbm Algorithm
params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'metric': 'mae',
    'num_leaves': 10,
    'verbose': 0
}
trn_data  = lgb.Dataset(X_train, label=y_train)
val_data  = lgb.Dataset(X_val, label=y_val)
#Check for MAE in the output Console 
model  = lgb.train(params, trn_data, 30000, valid_sets = [trn_data,val_data], verbose_eval=1000, early_stopping_rounds=1000)
pre_test_lgb  = model.predict(test_without_income)

#Saving predicted Values to Output file
outputIncome = pd.DataFrame(test['Instance'])
outputIncome['Income'] = pd.DataFrame(pre_test_lgb)
outputFileName = 'tcd-ml-1920-group-income-submission.csv'
outputIncome.to_csv(outputFileName)
    
