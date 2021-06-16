from __future__ import division, print_function

import os
import sys
import logging
import csv
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from xgboost import plot_importance
from sklearn import model_selection
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager
# font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
# font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

import warnings
warnings.filterwarnings("ignore")

## """ Read the data from the provided task_file """
task_file = ('./task_data.csv')
with open(task_file,mode='rb') as f:
    dataset = pd.read_csv(f)
# print(dataset.info())

## Check the data properties i.e. Identifying the important features within the provided data
## Benifits :
## 1.) help to concentrate on important features thus helping in building the model
## 2.) Get rid of features which are irrelavent and will not contribute to the model performance.
# print(dataset.describe())

## Split data into training features and labels

## Extracting Labels from the data
Y = dataset['class_label']
## Extracting Features from the data
dataset.drop(['sample index','class_label'],axis=1,inplace=True)
X = dataset

## Further Creating a Correlation Matrix to check out the correlation between the sensors in the dataset
corr_matrix = dataset.corr()

## Heatmap of the correlation matrix
sns.heatmap(corr_matrix, cmap='Blues', annot=True)
plt.savefig('./Correlation_Matrix of Sensors.png',dpi=300)
# plt.show()
plt.close()

## Defining the model to to rank the sensors according to their importance/predictive power
## with respect to the class labels of the samples
model = xgb.XGBClassifier()

## Fiting the model with data
model.fit(X, Y)

# make predictions for test set
Y_pred = model.predict(X)
predictions = [round(value) for value in Y_pred]

## Model Accuracy i.e how correct is the classifier?
accuracy = accuracy_score(Y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

## Visualizing the importance of each feature/sensor in the data set with their F-values.
plot_importance(model)
plt.savefig('./Feature_Fvalue_plot.png',dpi=300)
# plt.show()
plt.close()

## Ranking of sensors based on their importance value in a reverse order
rank_sensors = sorted(zip(model.feature_importances_, X.columns), reverse=True)
for label in rank_sensors:
    print('For {} Predicitve Score is {}'.format(label[1], label[0]))

## Seperating features and their importance values
scores = list(zip(*rank_sensors))[0]
sensors = list(zip(*rank_sensors))[1]
sensors_list = np.arange(len(sensors))

## Save ranked list of sensors and their importance values in a csv file
ranked_sensors = './ranked_sensors.csv'
header = ['Sensor', 'Importance value']
with open(ranked_sensors,'w') as output:
    ## create the csv writer
    write = csv.writer(output)
    ## write the header
    write.writerow(header)
    # write the row to csv file
    write.writerows(zip(sensors, scores))
output.close()

## Plot features with respect to their importance value
sns.set()
sns.set_style("whitegrid")
fig = plt.figure(figsize=(8, 4))
# sns.set_style("ticks")
sns.barplot(sensors_list, np.array(scores), palette="flare")
plt.xticks(sensors_list, sensors)
plt.ylabel('Importance Score')
plt.title('XGBClassifier Model')
# plt.tight_layout()
plt.show()
plt.savefig('./Sensors_vs_Importance_value_plot',dpi=300)
plt.close()


## https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
## https://www.kaggle.com/dansbecker/xgboost
## Using XGBoost https://medium.com/analytics-vidhya/a-guide-to-underrated-machine-learning-algorithms-alternatives-to-decision-tree-and-random-forest-6e2f8336d4d5#:~:text=However%2C%20the%20alternatives%20I'm,for%20regular%20projects%20as%20well.

## Testing the Accuracy and Efficiency of model
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, eval_metric='logloss')

## Training model using training data set
model.fit(X_train, y_train, early_stopping_rounds=50,eval_set=[(X_train, y_train), (X_test, y_test)])

## Evaluate the predicitions
predictions = model.predict(X_test)

## Mean Absolute error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

## Model Accuracy i.e how correct is the classifier?
accuracy = accuracy_score(y_test, predictions)
print('Accuracy of Model is : %.2f%%'%(accuracy*100.0))

## Visualizing the importance of each feature/sensor in the data set with their F-values.
plot_importance(model)
# plt.show()
plt.close()


# ## Performance metric from the model
# results = model.evals_result()
# epochs = len(results['validation_0']['error'])
# x_axis = range(0,epochs)
#
# ## Plot Log Loss
# fig, ax =plt.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label="Train")
# ax.plot(x_axis, results['validation_1']['logloss'], label="Test")
# plt.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')
# plt.savefig('./LogLoss.png',dpi=300)
# plt.show()
# plt.close()
#
# ## Plot Classification Error
# fig, ax =plt.subplots()
# ax.plot(x_axis, results['validation_0']['error'], label="Train")
# ax.plot(x_axis, results['validation_1']['error'], label="Test")
# plt.legend()
# plt.ylabel('Classification Error')
# plt.title('XGBoost Classification Error')
# plt.savefig('./Classification Error.png',dpi=300)
# plt.show()
# plt.close()



## Defining the model to to rank the sensors according to their importance/predictive power
## with respect to the class labels of the samples
rank_sensors_model = sorted(zip(model.feature_importances_, X.columns), reverse=True)
for score_model in rank_sensors:
    print('For {} Predicitve Score is {}'.format(score_model[1], score_model[0]))

## Visualizing the importance of each feature/sensor in the data set with their F-values.
scores_model = list(zip(*rank_sensors_model))[0]
sensors_model = list(zip(*rank_sensors_model))[1]
sensors_model_list = np.arange(len(sensors_model))

## Save new ranked list of sensors and their importance values in a new csv file
ranked_sensors_model = './ranked_sensors_model.csv'
header = ['Sensor', 'Importance value']
with open(ranked_sensors_model,'w') as output:
    ## create the csv writer
    write = csv.writer(output)
    ## write the header
    write.writerow(header)
    # write the row to csv file
    write.writerows(zip(sensors_model, scores_model))
output.close()

sns.set()
sns.set_style("whitegrid")
fig = plt.figure(figsize=(8, 4))
# sns.set_style("ticks")
sns.barplot(sensors_model_list, np.array(scores_model), palette='crest')
plt.xticks(sensors_model_list, sensors_model)
plt.ylabel('Importance Score')
plt.title('XGBClassifier Model')
# plt.tight_layout()
plt.show()
plt.close()

## https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
## Scalabolity with respect to numbers of features
## Fitting model using each importance as threshold
thresholds = np.sort(model.feature_importances_)
for threshold in thresholds:
    ## Selecting feature based on threshold value
    select_feature = SelectFromModel(model, threshold = threshold, prefit = True)
    select_X_train = select_feature.transform(X_train)
    ## Train the model for each feature
    select_feature_model = xgb.XGBClassifier(eval_metric='logloss')
    select_feature_model.fit(select_X_train, y_train)
    ## Evaluating the model
    select_X_test = select_feature.transform(X_test)
    feature_prediciton = select_feature_model.predict(select_X_test)
    feature_prediciton = [round(value) for value in feature_prediciton]
    feature_accuracy = accuracy_score(y_test, feature_prediciton)
    print('Threshold = %.5f, n=%d, Accuracy : %.2f%%' % (threshold,select_X_train.shape[1],feature_accuracy*100.0))
