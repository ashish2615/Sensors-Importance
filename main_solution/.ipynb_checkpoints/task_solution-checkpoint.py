from __future__ import division, print_function

import csv
import warnings
import numpy as np
import pandas as pd
from xgboost import plot_importance, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

warnings.filterwarnings("ignore")

## Read the data from the provided task_file.
## https://pandas.pydata.org/
task_file = ('./task_data.csv')
with open(task_file,mode='rb') as f:
    dataset = pd.read_csv(f)

## Check the data properties i.e. Identifying the important features within the provided data
## Benifits :
## 1.) help to concentrate on important features thus helping in building the model
## 2.) Get rid of features which are irrelavent and will not contribute to the model performance.
print(dataset.info())
print(dataset.describe())

## Split data into training features and labels
## Extracting Labels from the data
Y = dataset['class_label']

## Extracting Features from the data
dataset.drop(['sample index','class_label'],axis=1,inplace=True)
X = dataset

## Creating a Correlation Matrix to check out the correlation between the sensors in the dataset
## https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
corr_matrix = dataset.corr()

## Heatmap of the correlation matrix
## https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(corr_matrix, cmap='Blues', annot=True)
plt.savefig('./Correlation_Matrix of Sensors.png',dpi=300)
plt.show()
plt.close()

## split data into train and test dataset
## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

## Defining the model to rank the sensors according to their importance/predictive power
## with respect to the class labels of the samples
## Testing the Accuracy and Efficiency of model

## https://www.kaggle.com/dansbecker/xgboost
## https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
## https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

## Instantiate model
model = XGBClassifier(n_estimators=300, learning_rate=0.1)

## Training model using training data set
model.fit(X_train, y_train, eval_metric=['error','logloss'], early_stopping_rounds=10,
          eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

# print('Model Feature Importance', model.feature_importances_)
plt.bar(range(len(model.feature_importances_)),model.feature_importances_)
plt.xlabel('Sensors')
plt.ylabel('Importance')
plt.savefig('./Model_Feature_Plot',dpi=300)
plt.show()
plt.close()

## Visualizing the importance of each feature/sensor in the data set with their F-values.
plot_importance(model,importance_type='gain')
plt.savefig('./Feature Importance Plot',dpi=300)
plt.show()
plt.close()

## Ranking of sensors based on their importance value in a reverse order
rank_sensors = sorted(zip(model.feature_importances_, X.columns), reverse=True)
for label in rank_sensors:
    print('For {} Predictive Score is {}'.format(label[1], label[0]))

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

## Defining a function to plot the features vs importance score
## https://seaborn.pydata.org/generated/seaborn.barplot.html
def plot_features(features_list, importance_list, model_name, palette=None):
    sns.set()
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(features_list, np.array(importance_list), palette=palette)
    plt.xticks(sensors_list, sensors)
    plt.ylabel('Importance Score')
    plt.title(model_name)
    # plt.tight_layout()
    plt.savefig('./'+model_name+'_Sensors_vs_Importance_value_plot',dpi=300)
    plt.show()
    plt.close()

## Plot features with respect to their importance value
plot_features(sensors_list, scores, 'XGBoost Classifier', palette='flare')

## Evaluate the predictions
predictions = model.predict(X_test)

## Mean Absolute error
## https://scikit-learn.org/stable/modules/model_evaluation.html
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

## Model Accuracy i.e how correct is the classifier?
accuracy = accuracy_score(y_test, predictions)
print('Accuracy of Model is : %.2f%%'%(accuracy*100.0))

## https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
## Scalability with respect to numbers of features
## Fitting model using each importance as threshold
thresholds = np.sort(model.feature_importances_)
for threshold in thresholds:
    ## Selecting feature based on threshold value
    select_feature = SelectFromModel(model, threshold = threshold, prefit = True)
    select_X_train = select_feature.transform(X_train)

    ## Train the model for each feature
    select_feature_model = XGBClassifier(eval_metric='logloss')
    select_feature_model.fit(select_X_train, y_train)

    ## Evaluating the model
    select_X_test = select_feature.transform(X_test)

    ## Evaluating the predictions
    feature_prediction = select_feature_model.predict(select_X_test)
    feature_prediction = [round(value) for value in feature_prediction]

    ## Evaluating the Accuracy
    feature_accuracy = accuracy_score(y_test, feature_prediction)
    print('Threshold = %.5f, n=%d, Accuracy : %.2f%%' % (threshold,select_X_train.shape[1],feature_accuracy*100.0))


### Alternative Methods to Solve the Problem ###

## First Model: Random Forest
## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

## Extract Feature Importance Value and Rank them in a reverse order
rank_rf = sorted(zip(random_forest.feature_importances_, X.columns), reverse=True)
for label in rank_rf:
    print('For RandomForest {} Predictive Score is {}'.format(label[1], label[0]))

## Separating features and their importance values
scores_rf = list(zip(*rank_rf))[0]
sensors_rf = list(zip(*rank_rf))[1]
sensors_rf_list = np.arange(len(sensors_rf))

## Plot features with respect to their importance value
plot_features(sensors_rf_list, scores_rf, 'RandomForest Classifier', palette='crest')

## Evaluate the predictions
rf_predictions = random_forest.predict(X_test)

## Mean Absolute error
print("Mean Absolute Error : " + str(mean_absolute_error(rf_predictions, y_test)))

## Model Accuracy i.e how correct is the classifier?
rf_accuracy = accuracy_score(y_test, rf_predictions)
print('Accuracy of Random Forest Model is : %.2f%%'%(rf_accuracy*100.0))


## Second Model :  AdaBoost
## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
adaboost = AdaBoostClassifier()
adaboost.fit(X_train,y_train)

## Extract Feature Importance Value and Rank them in a reverse order
rank_ada = sorted(zip(adaboost.feature_importances_, X.columns), reverse=True)
for label in rank_ada:
    print('For AdaBoost {} Predictive Score is {}'.format(label[1], label[0]))

## Separating features and their importance values
scores_ada = list(zip(*rank_ada))[0]
sensors_ada = list(zip(*rank_ada))[1]
sensors_ada_list = np.arange(len(sensors_ada))

## Plot features with respect to their importance value
plot_features(sensors_ada_list, scores_ada, 'AdaBoost Classifier', palette='dark:salmon_r')

## Estimating the performance of AdaBooster Classifier
ada_prediction = adaboost.predict(X_test)

## Mean Absolute error
print("Mean Absolute Error : " + str(mean_absolute_error(ada_prediction, y_test)))

## Model Accuracy i.e how correct is the classifier?
ada_accuracy = accuracy_score(y_test, ada_prediction)
print('Accuracy of AdaBoost Model is : %.2f%%' % (ada_accuracy*100.0))