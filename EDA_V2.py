## Load Library 
# Import the neccessary library for data analysis
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Load Data
# Read the data from the csv file amd display the head of the dataframe

df_seq=pd.read_csv("pdb_data_seq.csv")
df_no_dups=pd.read_csv("pdb_data_no_dups.csv")
df_seq.head(6)
df_no_dups.head(6)
df_seq.info()
df_no_dups.info()


# ## Merge two dataset together
# Some column are duplicated during the merge, hence, the duplicated columns will be dropped. 
# Year column is not importnace in predicting the protein, hence it will be dropped.

df_merge_2 = df_no_dups.merge(df_seq, how='inner', on='structureId')
df_merge_2.info()
df_merge_2.macromoleculeType_y.head()
df_merge_2.macromoleculeType_x.head()
df2 = df_merge_2.rename({'macromoleculeType_x': 'macromoleculeType'}, axis=1)
df2 = df_merge_2.drop('macromoleculeType_y', 1)
df2 = df2.drop('residueCount_y', 1)
df2 = df2.rename({'residueCount_x': 'residueCount'}, axis=1)
df2 = df2.rename({'macromoleculeType_x': 'macromoleculeType'}, axis=1)
df2 = df2.drop('publicationYear', 1)
df2.columns
df2.info()

# Only the column with int and float data types are display 
df2.describe().T
# ## Checking for Null Value
# The null values will affect the accuracy of the model.
df2.isnull().any()
df2.isnull().sum()
def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
    :param series: Pandas DataFrame object
    :return: float
    """
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 2)

# Only include columns that contain any NaN values
df_with_any_null_values = df2[df2.columns[df2.isnull().any()].tolist()]

get_percentage_missing(df_with_any_null_values)

#Visualize the missing value
sns.heatmap(df2.isnull(), cbar=False)

# Remove the missing value
df_merge_clean = df2.dropna(how='any',axis=0) 

# Only include columns that contain any NaN values
df_with_any_null_values = df_merge_clean[df_merge_clean.columns[df_merge_clean.isnull().any()].tolist()]

# Double check whether the null values are removed or not
get_percentage_missing(df_with_any_null_values)
sns.heatmap(df_merge_clean.isnull(), cbar=False)

# Writing the clean dataset to csv file
df_merge_clean.to_csv("pdb_merge_clean.csv", index=False)

# Shape before dropping the null value
print(f'The shape of the dataframe before dropping the null value {df2.shape}')

# Shape after dropping the null value
print(f'The shape of the dataframe after dropping the null value {df_merge_clean.shape}')

# Checking the colums
df_merge_clean.columns



# ## Visualize the data

classfication = df_merge_clean["classification"].value_counts()
macromoleculeType =  df_merge_clean["macromoleculeType"].value_counts()
residueCount =  df_merge_clean["residueCount"].value_counts()
sequence =  df_merge_clean["sequence"].value_counts()
phValue =  df_merge_clean["phValue"].value_counts()


# ## Protein Classifcation
from collections import Counter

# count numbers of instances per class
cnt = Counter(df_merge_clean.classification)

# select only 10 most common classes
top_classes = 10
# sort classes
sorted_classes = cnt.most_common()[:top_classes]
# Create list
classes = [c[0] for c in sorted_classes] 
counts = [c[1] for c in sorted_classes]
print("at least " + str(counts[-1]) + " instances per class")

# apply to dataframe
print(str(df_merge_clean.shape[0]) + " instances before")
df_merge_clean = df_merge_clean[[c in classes for c in df_merge_clean.classification]]
print(str(df_merge_clean.shape[0]) + " instances after")

popular_protein_class = df_merge_clean.classification.value_counts()[:10] # Extract the top 10 Protein Tech 
ppl_pro_class_df = pd.DataFrame(popular_protein_class).reset_index()
ppl_pro_class_df.columns=['Protein Classification','values']

ppl_pro_class_df.head(10)

## Bar chart
f,ax = plt.subplots(figsize=(10,8))
ppl_pro_class_df.plot(kind = 'barh',ax=ax,color='gray',legend=None,width= 0.8)
# get_width pulls left or right; get_y pushes up or down
for i in ax.patches:
    ax.text(i.get_width()+.1, i.get_y()+.40, str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  
#Set ylabel
ax.set_yticklabels(ppl_pro_class_df['Protein Classification'])
# invert for largest on top 
ax.invert_yaxis()
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
ax.tick_params(**kwargs)
x_axis = ax.axes.get_xaxis().set_visible(False)
ax.set_title ('Top 10 Protein Classification',color='black',fontsize=16)
sns.despine(bottom=True)


# ## Sequence
# count numbers of instances per class of sequence
seq_cnt = Counter(df_merge_clean.sequence)

# select only 10 most common classes!
top_classes = 10
# sort classes
seq_sorted_classes = seq_cnt.most_common()[:top_classes]
# Create list
seq_classes = [c[0] for c in seq_sorted_classes] 
seq_counts = [c[1] for c in seq_sorted_classes]
print("at least " + str(counts[-1]) + " instances per class")

sequence = df_merge_clean[[c in seq_classes for c in df_merge_clean.sequence]]

plt.bar(range(len(seq_classes)), seq_counts)
plt.xticks(range(len(seq_classes)), seq_classes, rotation='vertical')
plt.ylabel('frequency')
plt.xlabel('Protein sequence')
plt.show()



popular_seq = df_merge_clean.sequence.value_counts()[:5] # Extract the 5 top used Exp Tech 
ppl_seq_df = pd.DataFrame(popular_seq).reset_index()
ppl_seq_df.columns=['Protein Sequence','values']


ppl_seq_df.head(10)
labels = ppl_seq_df['Protein Sequence']
values = ppl_seq_df['values']

# Plot graph
# Pie Chart
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                  textprops=dict(color="w"))

ax.legend(wedges, labels,
          title="Classification",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Sequence in Percentage")
plt.show()



f,ax = plt.subplots(figsize=(10,8))
ppl_seq_df.plot(kind = 'barh',ax=ax,color='gray',legend=None,width= 0.8)
# get_width pulls left or right; get_y pushes up or down
for i in ax.patches:
    ax.text(i.get_width()+.1, i.get_y()+.40, str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  
#Set ylabel
ax.set_yticklabels(ppl_seq_df['Protein Sequence'])
# invert for largest on top 
ax.invert_yaxis()
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
ax.tick_params(**kwargs)
x_axis = ax.axes.get_xaxis().set_visible(False)
ax.set_title ('Top 5 Protein Sequence',color='black',fontsize=16)
sns.despine(bottom=True)


# # Ph of The Protein 
# pH > 7 == alkaline
# pH < 7 == acid
# create function to calculate the pH of the protein

def ph_scale (ph):
    if ph < 7 :
        ph = 'Acidic'
    elif ph > 7:
        ph = 'Alkaline'
    else:
        ph = 'Neutral'
    return ph

print('The pH Scale are group into 3 Categories: BASIC if [ pH > 7 ], ACIDIC if [ pH < 7 ] and NEUTRAL if pH [ is equal to 7 ]')
df_merge_clean['pH_Scale'] = df_merge_clean["phValue"].apply(ph_scale)
df_merge_clean.head()

# Graph
# Data
labels= df_merge_clean['pH_Scale'].value_counts().index
values = df_merge_clean['pH_Scale'].value_counts().values


# Plot graph
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                  textprops=dict(color="w"))
ax.legend(wedges, labels,
          title="Classification",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
ax.set_title("pH Distribution in Percentage")
plt.show()



# ## Common Crystalise Technique
popular_crys_tech = df_merge_clean.crystallizationMethod.value_counts()[:5] # Extract the 5 top used Exp Tech 
ppl_crys_tech_df = pd.DataFrame(popular_crys_tech).reset_index() # Create a dataframe
ppl_crys_tech_df.columns=['Crystalize Technique','values'] # Add the column name

ppl_crys_tech_df.head(10)
# Graph
# Data
labels= ppl_crys_tech_df['Crystalize Technique']
values = ppl_crys_tech_df['values']


# Plot graph
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                  textprops=dict(color="w"))
# wedges, texts = ax.pie(values,
#                                   textprops=dict(color="w"))

ax.legend(wedges, labels,
          title="Technique",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Most Used Crystalize Technque")
plt.show()


# ## MacromoleculeType
popular_mole_type = df_merge_clean.macromoleculeType.value_counts()[:5] # Extract the 5 top used Exp Tech 
ppl_mole_type_df = pd.DataFrame(popular_mole_type).reset_index() # Create a dataframe
ppl_mole_type_df.columns=['macromoleculeType','values'] # Add the column name

ppl_mole_type_df.head(10)
# Graph
# Data
labels= ppl_mole_type_df['macromoleculeType']
values = ppl_mole_type_df['values']


fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                  textprops=dict(color="w"))
# wedges, texts = ax.pie(values,
#                                   textprops=dict(color="w"))

ax.legend(wedges, labels,
          title="Type of Molecule",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Most Common MacroMolelcule in the Protein")
plt.show()



f,ax = plt.subplots(figsize=(10,8))
ppl_mole_type_df.plot(kind = 'barh',ax=ax,color='gray',legend=None,width= 0.8)
# get_width pulls left or right; get_y pushes up or down
for i in ax.patches:
    ax.text(i.get_width()+.1, i.get_y()+.40,             str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  
#Set ylabel
ax.set_yticklabels(ppl_mole_type_df['macromoleculeType'])
# invert for largest on top 
ax.invert_yaxis()
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
ax.tick_params(**kwargs)
x_axis = ax.axes.get_xaxis().set_visible(False)
ax.set_title ('Top 5 MacroMolecule',color='black',fontsize=16)
sns.despine(bottom=True)

## Experiment technique
popular_experimentalTechnique = df_merge_clean.experimentalTechnique.value_counts()[:5] # Extract the 5 top used Exp Tech 
ppl_experimentalTechnique_df = pd.DataFrame(popular_experimentalTechnique).reset_index() # Create a dataframe
ppl_experimentalTechnique_df.columns=['experimentalTechnique','values'] # Add the column name
print(ppl_experimentalTechnique_df.head(10))

# Graph
# Data
labels= ppl_experimentalTechnique_df['experimentalTechnique']
values = ppl_experimentalTechnique_df['values']

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct)

wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                  textprops=dict(color="w"))
ax.legend(wedges, labels,
          title="Popular Experiment Technique",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Most Common Experiment Technique")
plt.show()


popular_densityMatthews = df_merge_clean.densityMatthews.value_counts()[:5] # Extract the 5 top used Exp Tech 
ppl_densityMatthews_df = pd.DataFrame(popular_densityMatthews).reset_index() # Create a dataframe
ppl_densityMatthews_df.columns=['densityMatthews','values'] # Add the column name
ppl_densityMatthews_df.head(10)
ppl_densityMatthews_df.describe()


# ## Correlation & Coefficient

df_merge_clean_numeric = df_merge_clean.select_dtypes(include=['float64', 'int64'])
df_merge_clean_numeric.head()
df_merge_clean_numeric.corr()
df_merge_clean_numeric.cov()









