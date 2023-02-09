# RandomForestRegression4Kinetics
Hello everyone.

There are two notebooks to show how you can train a Random Forest Regression to predict activation energies.
A third notebook is included showing you how to generate your own datasets.

We provide 2 separate notebooks “RFR_sklearn.ipynb”  and  “RFR_xgboost.ipynb” which train the SKLearn models and XGBoost model
respectively. The training data used is contained in a single JSON file “all_data.json”. In this file we contain the total list
of reactions with an InChI-key of every reagent as well as some information on the reaction under the key “reaction_list”.

A third notebook shows how to generate training sets. Meaning you can add/remove features on include new reactions
to train on.

This requires a python installation with the following packages
XGBoost
scikit-learn
matplotlib
tqdm
pandas
numpy
shutil
rdkit
jupyter (to run the notebooks)

for any issues/suggestions/comments message I will be happy to discuss and implement changes!
ealonso1995@gmail.com
