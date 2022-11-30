# RandomForestRegression4Kinetics
Hello everyone.

This are two notebooks to show how you can train a Random Forest Regression to predict activation energies.
The two notebooks show how to do this with SKLearn and XG Boost

We provide 2 separate notebooks “RFR_sklearn.ipynb”  and  “RFR_xgboost.ipynb” which train the SKLearn models and XGBoost model
respectively. The training data used is contained in a single JSON file “all_data.json”. In this file we contain the total list
of reactions with an InChI-key of every reagent as well as some information on the reaction under the key “reaction_list”.

This requires a python installation with the following packages
scikit-learn
matplotlib
tqdm
pandas
numpy
shutil
jupyter (to run the notebooks)

for any issues message 
ealonso1995@gmail.com
