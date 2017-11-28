'''
Created on 6 Aug 2017

@author: gopora
'''
import piu_functions as piuf
import pandas as pd

trv, trl = piuf.piu_load_data()

trv["funder"].fillna("Other_NA", inplace=True)
trv["funder"].replace(to_replace="0", value="Other_0", inplace=True)

#print (trv['funder'].value_counts())

bins = [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 9000, 10000]
group_names = ['ten', 'hundred', 'twoh', 'threeh', 'fourh', 'fiveh', 'sixh', 'sevenh', 'eighth', 'nineh', 'thousand', 'twoth', 'threeth', 'nineth', 'tenth']

trv["funder_count"] = trv.groupby("funder")["funder"].transform("count")

#print (trv["funder_count"].value_counts())

#trv['funder_cat'] = pd.cut(trv['funder_count'], bins, labels=group_names)

#print (trv["funder_cat"])

#print(trv[trv["funder_cat"].isnull()])

def tresen(x):
    if (x["funder"] != "Other_NA" and x["funder"] != "Other_0"):
        if (x['funder_count'] > 5000):
            return "**"
        else:
            return "*"
    else:
        return x["funder"]
    
trv["funder_cat"] = trv[['funder', 'funder_count']].apply(tresen, axis=1)

print (trv)