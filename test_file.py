from __future__ import division, print_function

import csv
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, XGBClassifier
import eli5
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from eli5.sklearn import PermutationImportance
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

warnings.filterwarnings("ignore")

## """ Read the data from the provided task_file """
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

## Further Creating a Correlation Matrix to check out the correlation between the sensors in the dataset
corr_matrix = dataset.corr()

## Heatmap of the correlation matrix
## https://seaborn.pydata.org/
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

## instantiate model
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


def plot_importance(model, importance_type=None):
    """
    Visualizing the importance of each feature/sensor in the data set with their F-values.
    model : Model used
    importance_type : str
        Type of importance used to plot the features : weight, gain, control
    return
        Plot of model features with their importance value and F-score
    """
    ## Visualizing the importance of each feature/sensor in the data set with their F-values.
    plot_importance(model,importance_type='gain')
    plt.savefig('./Feature Importance Plot',dpi=300)
    plt.close()

def rank_features(model_values, feature_name, reverse=None):
    """
    ## Ranking of sensors based on their importance value in a reverse order ##
    param model_vals : float
        Values of Importance corresponding to each feature
    param feat_name: str
        Name of Each feature
    param reverse: Bool
        To arrange the features in reverse order
    return: list
        List of features and their corresponding importance values.
    """

    ## Ranking of sensors based on their importance value in a reverse order
    rank = sorted(zip(model_values, feature_name), reverse=reverse)
    for label in rank_rf:
        print('For {} Predicitve Score is {}'.format(label[1], label[0]))

    return rank

def save_rank(feature_name, importance_value):
    """

    return : data file
        returns csv data file of ranked features with their importance values
    """
    ## Save ranked list of sensors and their importance values in a csv file
    filename = './ranked_sensors.csv'
    header = ['Sensor', 'Importance value']
    with open(ranked_sensors, 'w') as output:
        ## create the csv writer
        write = csv.writer(output)
        ## write the header
        write.writerow(header)
        # write the row to csv file
        write.writerows(zip(sensors, scores))
    output.close()

## Seperating features and their importance values
scores = list(zip(*rank_sensors))[0]
sensors = list(zip(*rank_sensors))[1]
sensors_list = np.arange(len(sensors))

## Plot features with respect to their importance value
sns.set()
sns.set_style("whitegrid")
fig = plt.figure(figsize=(8, 4))
sns.barplot(sensors_list, np.array(scores), palette="flare")
plt.xticks(sensors_list, sensors)
plt.ylabel('Importance Score')
plt.title('XGBClassifier Model')
# plt.tight_layout()
plt.savefig('./Sensors_vs_Importance_value_plot',dpi=300)
plt.show()
plt.close()

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

## Evaluate the predicitions
predictions = model.predict(X_test)

## Mean Absolute error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

## Model Accuracy i.e how correct is the classifier?
accuracy = accuracy_score(y_test, predictions)
print('Accuracy of Model is : %.2f%%'%(accuracy*100.0))

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

    ## Evaluating the predicition
    feature_predicition = select_feature_model.predict(select_X_test)
    feature_predicition = [round(value) for value in feature_predicition]

    ## Evaluating the Accuracy
    feature_accuracy = accuracy_score(y_test, feature_predicition)
    print('Threshold = %.5f, n=%d, Accuracy : %.2f%%' % (threshold,select_X_train.shape[1],feature_accuracy*100.0))


""" Alternative Methods to Solve the Problem"""

## First Model: Random Forest
## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

## Extract Feature Importance Value and Rank them in a reverse order
rank_rf = sorted(zip(random_forest.feature_importances_, X.columns), reverse=True)
for label in rank_rf:
    print('For {} Predicitve Score is {}'.format(label[1], label[0]))

## Seperating features and their importance values
scores_rf = list(zip(*rank_rf))[0]
sensors_rf = list(zip(*rank_rf))[1]
sensors_rf_list = np.arange(len(sensors_rf))

## Plot features with respect to their importance value
sns.set()
sns.set_style("whitegrid")
fig = plt.figure(figsize=(8, 4))
sns.barplot(sensors_rf_list, np.array(scores_rf), palette="crest")
plt.xticks(sensors_rf_list, sensors_rf)
plt.ylabel('Importance Score')
plt.title('RandomForest Classifier Model')
# plt.tight_layout()
plt.savefig('./Sensors_vs_Importance_RF_plot',dpi=300)
plt.show()
plt.close()

## Evaluate the predicitions
rf_predictions = random_forest.predict(X_test)

## Mean Absolute error
print("Mean Absolute Error : " + str(mean_absolute_error(rf_predictions, y_test)))

## Model Accuracy i.e how correct is the classifier?
rf_accuracy = accuracy_score(y_test, rf_predictions)
print('Accuracy of Model is : %.2f%%'%(rf_accuracy*100.0))


## Second Model :  AdaBoost
## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
adaboost = AdaBoostClassifier()
adaboost.fit(X_train,y_train)

## Extract Feature Importance Value and Rank them in a reverse order
rank_ada = sorted(zip(adaboost.feature_importances_, X.columns), reverse=True)
for label in rank_ada:
    print('For {} Predicitve Score is {}'.format(label[1], label[0]))

## Seperating features and their importance values
scores_ada = list(zip(*rank_ada))[0]
sensors_ada = list(zip(*rank_ada))[1]
sensors_ada_list = np.arange(len(sensors_ada))

## Plot features with respect to their importance value
sns.set()
sns.set_style("whitegrid")
fig = plt.figure(figsize=(8, 4))
sns.barplot(sensors_ada_list, np.array(scores_ada), palette='dark:salmon_r')
plt.xticks(sensors_list, sensors)
plt.ylabel('Importance Score')
plt.title('AdaBoost Classifier Model')
# plt.tight_layout()
plt.savefig('./Sensors_vs_Importance_AB_plot',dpi=300)
plt.show()
plt.close()

## Estimating the performace of AdaBooster Clasifier
ada_prediction = adaboost.predict(X_test)

## Mean Absolute error
print("Mean Absolute Error : " + str(mean_absolute_error(ada_prediction, y_test)))

## Model Accuracy i.e how correct is the classifier?
ada_accuracy = accuracy_score(y_test, ada_prediction)
print('Accuracy of Model is : %.2f%%'%(ada_accuracy*100.0))





# ## First Model : XGBoost
# def xgb_model():
#     return XGBClassifier()
#
# ## Second Model : Random Forest
# def rf_model():
#     return RandomForestClassifier()
#
# ## Third Model : AdaBoost
# def ada_model():
#     return AdaBoostClassifier()






