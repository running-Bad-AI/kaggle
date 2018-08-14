import pandas as pd
import re
import matplotlib as plt
import numpy as np
from datetime import *

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split

# Read the dataset
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

dataset = train.drop(['AnimalID', 'OutcomeSubtype', 'OutcomeType'], axis=1)
dataset = dataset.append(test.drop('ID', axis=1), ignore_index=True)


# print(dataset)
print(train.shape, test.shape, dataset.shape)  #(26729, 10) (11456, 8) (38185, 7)
# print(dataset.isnull().sum())
# print(dataset.info())


# calculate Age in days
def calculate_age(x):
    if pd.isnull(x):
        return x
    num = int(x.split(' ')[0])
    if 'year' in x:
        return num * 365
    elif 'month' in x:
        return num * 30
    elif 'week' in x:
        return num * 7


def has_name(x):
    if pd.isnull(x):
        return 0
    return 1


def is_mix(x):
    if 'Mix' in x:
        return 1
    return 0


# data transformation


# AgeuponOutcome
dataset['AgeuponOutcome'] = dataset['AgeuponOutcome'].apply(lambda x: calculate_age(x))
dataset['AgeuponOutcome'].fillna(dataset['AgeuponOutcome'].dropna().mean(), inplace=True)

# SexuponOutcome
# Since there is only one NA, I will assign it to maximum class
dataset['SexuponOutcome'].fillna('Neutered Male', inplace=True)

# Name --> HasName, Name
# Does Animal has a name
dataset['HasName'] = dataset['Name'].apply(has_name)
dataset['Name'].fillna('Unknown', inplace=True)

# SexuponOutcome --> Sterilized, Sex
# Break SexuponOutcome into two - Sterilized and Sex
sex = dataset['SexuponOutcome'].str.split(' ', expand=True)
dataset['Sterilized'] = sex[0]
dataset['Sterilized'].fillna('Unknown', inplace=True)
dataset['Sex'] = sex[1]
dataset['Sex'].fillna('Unknown', inplace=True)

# Datetime --> Year, Month, Day, Hour
dates = dataset['DateTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
dataset['Year'] = dates.apply(lambda x: x.year)
dataset['Month'] = dates.apply(lambda x: x.month)
dataset['Day'] = dates.apply(lambda x: x.weekday())
dataset['Hour'] = dates.apply(lambda x: x.hour)

# Breed --> IsMix, Breed_1, Breed_2, Multiple_Breeds
# Is animal of mix breed?
dataset['IsMix'] = dataset['Breed'].apply(is_mix)
dataset['Breed_New'] = dataset['Breed'].apply(lambda x: x.split(' Mix')[0])
breeds = dataset['Breed_New'].apply(lambda x: x.split('/'))
dataset['Breed_1'] = breeds.apply(lambda x: x[0])
dataset['Breed_2'] = breeds.apply(lambda x: 'Unknown' if len(x) == 1 else x[1])
dataset['Multiple_Breeds'] = dataset['Breed'].apply(lambda x: 1 if '/' in x else 0)

# Color --> Color_1, Color_2, Multiple_Colors
colors = dataset['Color'].apply(lambda x: x.split('/'))
dataset['Color_1'] = colors.apply(lambda x: x[0])
dataset['Color_2'] = colors.apply(lambda x: 'Unknown' if len(x) == 1 else x[1])
dataset['Multiple_Colors'] = dataset['Color'].apply(lambda x: 1 if '/' in x else 0)

# # Drop unnecessary columns
drop_columns = ['DateTime', 'SexuponOutcome', 'Breed', 'Color', 'Breed_New']
dataset = dataset.drop(drop_columns, axis=1)

# Encoding categorical columns
enc = LabelEncoder()
cols = ['Name', 'AnimalType', 'Sterilized', 'Sex', 'Year', 'Month', 'Breed_1', 'Breed_2', 'Color_1', 'Color_2']
for col in cols:
    dataset[col] = enc.fit_transform(dataset[col])
# print(dataset)
# dataset.to_csv('tmp2.csv', sep=" ", index=False)


# save featured data to csv
train_x = dataset.loc[0:26728, ]  # 注意这里的冒号是左闭右闭
train_y = pd.DataFrame(enc.fit_transform(train['OutcomeType']), columns=['label'])
train_data = pd.concat([train_x, train_y], axis=1)
# print(data)
print(enc.classes_) # ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'] 默认按首字母排序
# train_test_split
train, validation = train_test_split(train_data.values, test_size=0.3)
test = dataset.loc[26729:, ].astype('int')
# 给test最后增加一列label,保持和train一样，否则解析会有问题
test["label"] = 1

train = pd.DataFrame(train, dtype=int)
validation = pd.DataFrame(validation, dtype=int)
train.to_csv('./train_data/train/featured_train.csv', sep=',', index=False, header=False)
validation.to_csv('./train_data/validation/featured_validation.csv', sep=',', index=False, header=False)
test.to_csv('./train_data/test/featured_test.csv', sep=',', index=False, header=False)

print(dataset.describe())
print(dataset.columns)
'''
                Name    AnimalType  AgeuponOutcome       HasName    Sterilized  \
min        0.000000      0.000000        0.000000      0.000000      0.000000   
max     7968.000000      1.000000     8030.000000      1.000000      3.000000   

                Sex          Year         Month           Day          Hour  \
min        0.000000             0      0.000000      0.000000      0.000000   
max        2.000000             3     11.000000      6.000000     23.000000   

              IsMix       Breed_1       Breed_2  Multiple_Breeds  \
min        0.000000      0.000000      0.000000         0.000000   
max        1.000000    230.000000    158.000000         1.000000   

            Color_1       Color_2  Multiple_Colors  
min        0.000000      0.000000         0.000000  
max       56.000000     46.000000         1.000000  
'''