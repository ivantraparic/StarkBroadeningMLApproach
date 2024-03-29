If you use this database, please cite:

Tapalaga, I., Traparic, I., Trklja Boca, N., et al. 2022, Neural Comput. Applic., 34, 6349

# StarkBroadeningMLApproach
Predicting Stark Broadening of a line using ML algorithm

Complete code is given in the file MLModel.py 

Our final dataset used is given in MLData.csv

Folders containing all emiters are not given because there is too much files to be uploaded in one folder (around 527 files in each folder). They could be sent by author if asked as a zip file.

After importing of all libraries that are needed, we created a class TransitionFinding in order to find wanted spectral transition from database, to later show regularities on graph. This is the sole purpose of this class.

After definition of the class we proceeded with data cleaning, where the steps are clearly visible.

Next we used GridSearchCV to find optimal parameters and best model of 4 considered: Linear Regression, Decision Tree Regressor, Random Forest Regressor and Gradient Boosting Regressor.

After we concluded that Random Forest Regressor is our wining model with R^2 = 0.95 with n_estimators = 100, we trained the model and the model was prepared to make predictions.

Next step in code is import of transitions of element for which we want to predict Stark broadening width. Different sitations are predicted, namely if we want to analyse the spectral transitions where exists more than one multiplet, and also if we analyse some of our previous work (part with OldData). Also a part where we fix transition means that we select only one series (for example 3p-nd: that would include transitions: 3p-4d, 3p-5d, 3p-6d...)

Finally we make predictions of new widths with defined model, and the rest of the code there is just preparation of the graphs.

Additionaly we performed analysis to see whether the model will predict quantum structure of atomic transitions or not. That is the last part of the code. 


