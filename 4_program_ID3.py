import pandas as pd
import math
from collections import Counter
from pprint import pprint
data = pd.read_csv('3.csv')

def entropy(a_list):
    cnt = Counter(x for x in a_list)
#    print(cnt)
    num_instances = len(a_list)*1.0
#    print(num_instances)
    probs = [x / num_instances for x in cnt.values()]
#    print(probs)
    ent=sum( [-prob*math.log(prob, 2) for prob in probs] )
    return ent

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df) * 1.0
#    print(df_split.groups)
    df_agg_ent = df_split.agg({target_attribute_name : [entropy, lambda y: len(y)/nobs] })
#    print(df_agg_ent)
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy(df[target_attribute_name])
    return old_entropy - new_entropy

def id3(df, target_attribute_name, attribute_names):
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt) == 1:
        return next(iter(cnt))
    else:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        tree = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,target_attribute_name,remaining_attribute_names)
            tree[best_attr][attr_val] = subtree
        return tree
attribute_names = list(data.columns)
print("List of Attributes:", attribute_names)
attribute_names.remove('PlayTennis')
print("Predicting Attributes:", attribute_names)
total_entropy = entropy(data['PlayTennis'])
print("Entropy of given PlayTennis Data Set:",total_entropy)
tree = id3(data,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)