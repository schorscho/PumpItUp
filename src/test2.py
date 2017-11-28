'''
Created on 8 Aug 2017

@author: gopora
'''
import piu_functions as piuf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

trv, trl = piuf.piu_load_data()

trv["installer"].fillna("Other_NA", inplace=True)
trv["funder"].fillna("Other_NA", inplace=True)

print(trv["installer"])

lb = LabelBinarizer()

lb.fit_transform(trv[["installer"]])