
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#  pip install lightgbm for window/ linux or macOs -> sudo  pip install lightgbm 
import lightgbm as ltb
# ## CatBoost
import CatBoostClassifier as ctb
import catboost
from keras.models import Sequential
from keras.layers import Dense
import sklearn.metrics as skm 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


df = pd.read_csv("DF_Bows_Merge.csv")

## create df_protein

clasific =df.classification.value_counts(ascending=False)
df_class = pd.DataFrame(round(((clasific/df.shape[0])*100),2).head(10)).reset_index()
df_class.columns = ['Classification', 'percent_value']

df_class.Classification.values.tolist()[1:5]
# Reduce the df_merge to df_protein which is compose of macromolecule type [Protein, Protein#RNA and Protein#DNA]
macrotype = ['Protein','Protein#RNA', 'Protein#DNA']
protein = ["HYDROLASE", "TRANSFERASE", "OXIDOREDUCTASE", "LYASE", "IMMUNE SYSTEM", "HYDROLASE/HYDROLASE INHIBITOR",
           "RIBOSOME", "TRANSCRIPTION", "ISOMERASE", "LIGASE"]
df_protein = df[(df['experimentalTechnique'] =='X-RAY DIFFRACTION') & 
                      (df['macromoleculeType'].isin(macrotype))]

df_protein.reset_index(drop=True,inplace=True)
df_protein_1 = df_protein[df_protein['classification'].isin(protein)]

df_protein_1.reset_index(drop=True,inplace=True)


class_dict = {'HYDROLASE':1, 'TRANSFERASE':2,'OXIDOREDUCTASE':3, 'LYASE':4, "IMMUNE SYSTEM":5,
              'HYDROLASE/HYDROLASE INHIBITOR': 6, 'RIBOSOME':7,
             'TRANSCRIPTION': 8, 'ISOMERASE': 9, 'LIGASE': 10}
df_protein_1.loc[:, 'class'] = df_protein_1['classification'].map(class_dict)
print(df_protein_1.head())
print(df_protein_1.shape)
object_columns = df_protein_1.select_dtypes(['object']).columns
print(object_columns)

# Dropping the string columns
df_protein_1.drop(object_columns, inplace=True, axis=1)
print(df_protein_1.columns)
df_protein_1.dtypes
columns = ['phValue', 'crystallizationTempK']
df_protein_1.drop(columns, inplace=True, axis=1)

print(df_protein_1.columns)


df_ml = df_protein_1.copy()

# spliting the data

X = df_protein_1.drop('class',axis = 1)
y = df_protein_1['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=123)
# Standardizing the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)


# Train tge data with Lightgbm model
LBM = ltb.LGBMClassifier()
LBM.fit(X_train, y_train)

# Validation score for test set
print(LBM.score(X_test, y_test))
print(LBM.score(X_val, y_val))

# cross validation
scores = cross_val_score(LBM, X_test, y_test, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# validation data set
scores = cross_val_score(LBM, X_val, y_val, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Tuned parameter

LBM_1 = ltb.LGBMClassifier(num_leaves= 90, max_depth= -3, learning_rate= 0.1)
LBM_1.fit(X_train, y_train)


# Validation score for test set
LBM_1.score(X_test, y_test)

# Validation score for validation set
LBM_1.score(X_val, y_val)



scores = cross_val_score(LBM_1, X_test, y_test, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(LBM_1, X_val, y_val, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Decision Tree 
# Test dataset
# crossvalidation function is being used
dt= DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(dt, X_test, y_test, cv=10)
scores.mean()


# validation dataset
dt= DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(dt, X_val, y_val, cv=10)
scores.mean()

# RandomForest
# Test set
rcf = RandomForestClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(rcf, X_test, y_test, cv=10)
scores.mean()

# Validaion set
rcf = RandomForestClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(rcf, X_val, y_val, cv=10)
scores.mean()

# ExtraTreesClassifier
# Test dataset
etc = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(etc, X_test, y_test, cv=10)
scores.mean()

# Validation
etc = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(etc, X_val, y_val, cv=10)
scores.mean()

# Catboost
import catboost
from catboost import Pool

eval_dataset = Pool(X_test, y_test)

ctb = catboost.CatBoostClassifier(loss_function='MultiClass')
ctb.fit(X_train, y_train, eval_set=eval_dataset)


print(ctb.get_best_score())


# Test set
print(ctb.score(X_test, y_test))
print(ctb.score(X_val, y_val))
print(ctb.get_all_params())


print('Parameters currently in use:\n')
pprint(ctb.get_params())


## Tune
from catboost import Pool,CatBoostClassifier
params ={'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS']}

eval_dataset = Pool(X_test, y_test)

ctb = CatBoostClassifier(loss_function='MultiClass')

ctb_random = RandomizedSearchCV(estimator = ctb, param_distributions = params, 
                                cv = 5, verbose=2, scoring="accuracy")
ctb_random.fit(X_train, y_train)
print(ctb_random.best_params_)

# Second tuned
from sklearn.model_selection import  GridSearchCV

eval_dataset = Pool(X_test, y_test)

params = {'depth':[5, 7, 9],
          'iterations':[500, 1000, 1500],
          'learning_rate':[0.03,0.01,0.1, 1]}



ctb_first_tuned = CatBoostClassifier(iterations=1000,
                           bootstrap_type='MVS',
                           model_shrink_rate=0,
                           loss_function='MultiClass',
                           verbose=True)

ctb_search = GridSearchCV(estimator = ctb_first_tuned, param_grid = params, 
                                cv = 5, verbose=2, scoring="accuracy")

ctb_search.fit(X_train, y_train)

ctb_search.best_params_




# ### LightGBM
from sklearn.model_selection import RandomizedSearchCV

random_grid = {
    'num_leaves': [30, 50, 70, 80, 90],
    'boosting': ['gbdt', 'dart', 'goss'],
    'learning_rate': [0.001, 0.003, 0.01, 0.1, 1, 10],
    'max_depth': [-3, -2, -1, 1, 3, 5, 7],
    'n_estimators': [50, 100, 150, 200]
}

LBM_random = RandomizedSearchCV(estimator = LBM, param_distributions = random_grid, 
                             cv = 5, verbose=2, scoring="accuracy",n_iter=100)
LBM_random.fit(X_train, y_train)
LBM_random.best_params_


LGB = ltb.LGBMClassifier(num_leaves= 90, max_depth= -3, learning_rate= 0.1,  boosting='goss', n_estimators = 150)
LGB.fit(X_train, y_train)

print(LGB.score(X_test, y_test))
print(LGB.score(X_val, y_val))


# make predictions for test data
y_pred = LGB.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={
             'max_depth': [-7, -5, -3],
             'min_child_samples': sp_randint(20, 100), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
            }

gs = RandomizedSearchCV(
    estimator=LGB, param_distributions=param_test, 
    scoring='accuracy',
    cv=5, verbose=1
)



gs.fit(X_train, y_train)
print(gs.best_params_)


LGB = ltb.LGBMClassifier(num_leaves= 90, max_depth= -3, learning_rate= 0.1,  boosting='goss', n_estimators = 150, colsample_bytree = 0.750405543298072,
 min_child_samples= 390, min_child_weight= 1e-05, reg_alpha = 7, reg_lambda = 20, subsample = 0.9816006229423775)
LGB.fit(X_train, y_train)
print(LGB.score(X_test, y_test))
print(LGB.score(X_val, y_val))


from joblib import dump, load
dump(LGB, 'LGB.joblib') 
dump(LBM_1, 'LGB.joblib')


from sklearn.ensemble import AdaBoostClassifier
from pprint import pprint

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)

print(ada.score(X_test, y_test))
print('Parameters currently in use:\n')
pprint(ada.get_params())


# XGBoost
# XGBoost and Adaboost are bad in multiclassification. The reason can be c=found in the lightgbm literature review. 

from xgboost import XGBRegressor 
from sklearn.metrics import accuracy_score


xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import optimizers

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(64, input_dim=25, kernel_initializer='normal', activation='relu'))
	model.add(Dense(128, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
print(model.summary())
# model_1 = neural_network_1()
model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs = 15, batch_size=35, verbose = 1)


from keras.layers import LeakyReLU
def model_1():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=25, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(100, input_dim=25, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(250, input_dim=25, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(200, input_dim=25, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, kernel_initializer='normal', bias_initializer='zeros'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='rmsprop',  metrics=['accuracy'])
    return model

model_1 = model_1()
print(model_1.summary())

model_1.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs = 100, batch_size=75, verbose = 1)
# evaluate the model
scores = model_1.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model_1.metrics_names[1], scores[1]*100))

y_pred = model_1.predict(X_test)
model_1.evaluate(X_test, y_test)

# serialize model to JSON
model_1_json = model_1.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_1_json)
# serialize weights to HDF5
model_1.save_weights("model_1.h5")
print("Saved model to disk")


