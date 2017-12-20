import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys,os
import seaborn as sns
#%matplotlib inline


training_data = pd.read_csv(sys.argv[1])

#training_data.describe()

#Feature Transformations
#method to convert ages to bins called 'Unknown'for [-1,0], 'Baby' for [0-5],
#'Child' for [6-12], 'Teenager' for [13-19], 'Student' for [20-25], 'Young Adult' for [26-35], 'Adult' for [36-60],
#'Senior' for [61-100]
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 19, 25, 35, 60, 100)
    age_groups = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=age_groups)
    df.Age = categories
    return df

#method which simplifies the cabin feature by filling N/A values with 'N'
#also it takes only the first letter the cabin using splicing
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

#method to convert fares into bins using the numbers from .describe() statistics earlier
#the N/A values are filled with -0.5
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df



#Feature Extraction 
#the below method is used to extract two features from the Name column
#method to format the Name column to extract LName and NamePrefix
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df



#method to drop the features which we inconsider inconsequential
#we have selected ticket,Name,Embarked columns to be dropped due lack of variance or too many N/A values
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)


#this method calls all the above transformation methods one by one and applies it on our dataframe
def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df


transformed_train = transform_features(training_data)
#transformed_train.head()


#as we can see above our training data now has :
#1. LName, NamePrefix instead of 'Name' which has been dropped
#2. 'Ticket', 'Name', 'Embarked' have been dropped 
#3. 'Fare', 'Age' have been converted into convenient bins


#Now as a next step, we need to remember that machine learning algorithms need all the input to be numerical values
#But as we can observe from above 'Sex', 'Age' , 'Fare', 'Cabin', 'Lname', 'NamePrefix' are in nominal(string) format
#Hence we need to convert these into numerical values
#Here we proceed to use LabelEncoder from sklearn preprocessing to achieve 
#Every column in numerical form
from sklearn import preprocessing
def encode_features(df_train):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
    return df_train
    
data_train = encode_features(transformed_train)
#data_train.head()


#Now we proceed to test the various classifiers in Scikit-Learn to 
#see which classifiers works best on our data

#Below we are splitting up our training data into 80% training , 20% testing data 
# X contains all the columns except 'Survived' because that is the feature we predict
# Y consists only of the column 'Survived'
X = data_train.drop(['Survived'], axis=1)
Y = data_train.Survived
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 5)
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

print("Phase 1 of the project:")
#Our first classifier will be Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifiers = {}
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
Y_pred =  dtree.predict(X_test)
#print(confusion_matrix(Y_test,Y_pred))
print("Decision Tree accuracy:")
print(accuracy_score(Y_test,Y_pred))
classifiers["Decision Tree"]=dtree


# Artifical Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100), max_iter = 1000,alpha = 0.01, momentum = 0.7)
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)
nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Artificial Nueral Network:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["NeuralNetwork"]=clf


#Deep Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100,100,100), max_iter = 100,alpha = 0.3, momentum = 0.7,activation = "relu")
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)
nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Deep Neural Network:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["DeepNeuralNetwork"]=clf


#Support Vector Machine
clf = svm.SVC()
clf.set_params(C = 100, kernel = "rbf")
svm_clf = clf.fit(X_train,Y_train)
svm_predict = svm_clf.predict(X_test)
svm_acc = accuracy_score(Y_test,svm_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Support Vector Machines:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["SupportVectorMachine"]=clf


#Multinomial Naive Bayes
clf = MultinomialNB()
clf.set_params(alpha = 0.1)
nb_clf = clf.fit(X_train,Y_train)
nb_predict = nb_clf.predict(X_test)
nb_acc = accuracy_score(Y_test,nb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Multinomial Naive Bayes:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["MultinomialNaiveBayes"]=clf



#Logistic Regression
clf = LogisticRegression()
clf.set_params(C = 10, max_iter = 10)
lr_clf = clf.fit(X_train,Y_train)
lr_predict = lr_clf.predict(X_test)
lr_acc = accuracy_score(Y_test,lr_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Logistic Regression:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["LogisticRegression"]=clf



#k-NN Classifier
clf = KNeighborsClassifier()
clf.set_params(n_neighbors= 5,leaf_size = 30)
knn_clf = clf.fit(X_train,Y_train)
knn_predict = knn_clf.predict(X_test)
knn_acc = accuracy_score(Y_test,knn_predict)
param =  knn_clf.get_params()
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("k-NN :")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["KNN"]=clf


#Random Forest Classifier
clf = RandomForestClassifier()
clf.set_params(n_estimators = 500, max_depth = 100)
rf_clf = clf.fit(X_train,Y_train)
rf_predict = rf_clf.predict(X_test)
rf_acc = accuracy_score(Y_test,rf_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Random Forest Classifier:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["RandomForest"]=clf



#AdaBoost
clf = AdaBoostClassifier()
clf.set_params(n_estimators = 10, learning_rate = 0.5)
ada_clf = clf.fit(X_train,Y_train)
ada_predict = ada_clf.predict(X_test)
ada_acc = accuracy_score(Y_test,ada_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("AdaBoost:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["AdaBoost"]=clf



#Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.set_params(n_estimators = 100,learning_rate = 0.25)
gb_clf = clf.fit(X_train,Y_train)
gb_predict = gb_clf.predict(X_test)
gb_acc = accuracy_score(Y_test,gb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("GradientBoostingClassifier:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["GradientBoostingClassifier"]=clf




#Perceptron
clf = linear_model.Perceptron()
#clf.set_params(max_iter = 1000,alpha = 0.01)
pt_clf = clf.fit(X_train,Y_train)
pt_predict = pt_clf.predict(X_test)
pt_acc = accuracy_score(Y_test,pt_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Perceptron:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["Perceptron"]=clf



print("Next we proceed to apply Feature Scaling to see if the performance of our various classifiers improves")
#Feature scaling aims to bring the values of our numerical features between 0 and 1
#This is mainly done because large numerical values may skew our data and make the classifier weight it more


#this technique is known to improve the performance of classifiers using gradient descent such as neural nets,perceptron,etc


XX = data_train.drop(['Survived'], axis=1)
YY = data_train.Survived
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
XX[XX.columns] = scaler.fit_transform(XX[XX.columns])


#XX.head()


X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size = 0.20, random_state = 5)
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifiers = {}
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
Y_pred =  dtree.predict(X_test)
#print(confusion_matrix(Y_test,Y_pred))
print("Decision Tree accuracy:")
print(accuracy_score(Y_test,Y_pred))
classifiers["Decision Tree"]=dtree


# Artifical Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100), max_iter = 1000,alpha = 0.01, momentum = 0.7)
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)
nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Artificial Nueral Network:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["NeuralNetwork"]=clf



#Deep Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100,100,100), max_iter = 100,alpha = 0.3, momentum = 0.7,activation = "relu")
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)
nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Deep Neural Network:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["DeepNeuralNetwork"]=clf


#Support Vector Machine
clf = svm.SVC()
clf.set_params(C = 100, kernel = "rbf")
svm_clf = clf.fit(X_train,Y_train)
svm_predict = svm_clf.predict(X_test)
svm_acc = accuracy_score(Y_test,svm_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Support Vector Machines:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["SupportVectorMachine"]=clf



#Multinomial Naive Bayes
clf = MultinomialNB()
clf.set_params(alpha = 0.1)
nb_clf = clf.fit(X_train,Y_train)
nb_predict = nb_clf.predict(X_test)
nb_acc = accuracy_score(Y_test,nb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Multinomial Naive Bayes:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["MultinomialNaiveBayes"]=clf



#Logistic Regression
clf = LogisticRegression()
clf.set_params(C = 10, max_iter = 10)
lr_clf = clf.fit(X_train,Y_train)
lr_predict = lr_clf.predict(X_test)
lr_acc = accuracy_score(Y_test,lr_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Logistic Regression:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["LogisticRegression"]=clf



#k-NN Classifier
clf = KNeighborsClassifier()
clf.set_params(n_neighbors= 5,leaf_size = 30)
knn_clf = clf.fit(X_train,Y_train)
knn_predict = knn_clf.predict(X_test)
knn_acc = accuracy_score(Y_test,knn_predict)
param =  knn_clf.get_params()
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("k-NN :")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["KNN"]=clf



#Random Forest Classifier
clf = RandomForestClassifier()
clf.set_params(n_estimators = 500, max_depth = 100)
rf_clf = clf.fit(X_train,Y_train)
rf_predict = rf_clf.predict(X_test)
rf_acc = accuracy_score(Y_test,rf_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Random Forest Classifier:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["RandomForest"]=clf



#AdaBoost
clf = AdaBoostClassifier()
clf.set_params(n_estimators = 10, learning_rate = 0.5)
ada_clf = clf.fit(X_train,Y_train)
ada_predict = ada_clf.predict(X_test)
ada_acc = accuracy_score(Y_test,ada_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("AdaBoost:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["AdaBoost"]=clf



#Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.set_params(n_estimators = 100,learning_rate = 0.25)
gb_clf = clf.fit(X_train,Y_train)
gb_predict = gb_clf.predict(X_test)
gb_acc = accuracy_score(Y_test,gb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("GradientBoostingClassifier:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["GradientBoostingClassifier"]=clf



#Perceptron
clf = linear_model.Perceptron()
pt_clf = clf.fit(X_train,Y_train)
pt_predict = pt_clf.predict(X_test)
pt_acc = accuracy_score(Y_test,pt_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Perceptron:")
print("Accuracy"+"                "+"F-Score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["Perceptron"]=clf


from xgboost import XGBClassifier

clf = XGBClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("XG Boost:")
print("Accuracy"+"          "+"F-score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["XGBoost"]=clf


#Trying the VotingClassifier: tries to concetually combine different machine learning classifiers and use a majority
#vote to predict the labels.Such a classifier for a set of equally well performing classifiers in order to balance out
#their individual weaknesses.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier


clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])


clf1 = clf1.fit(X_train, Y_train)
clf2 = clf2.fit(X_train, Y_train)
clf3 = clf3.fit(X_train, Y_train)
eclf = eclf.fit(X_train, Y_train)

y_pred = eclf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

accuracy = cross_val_score(eclf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(eclf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Voting Classifier:")
print("Accuracy"+"        "+"F-score")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["VotingClassifier"]=eclf



#Here we print the performance of the various classifiers
print ("accuracy","              ","F-score")
for clf in classifiers.values():
    accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
    f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
    for i in classifiers:
        if classifiers[i]== clf:
            print (i),
            break
    print ( " : ",accuracy.mean(), "  ",f_score.mean())



#So from our results we can see that our best performing classifiers are:
#1)RandomForest:  0.835388441762    0.821362061256
#2)GradientBoostingClassifier:  0.821363179074    0.824240442656
#3)XGBoost:  0.815669572994    0.815669572994
#4)DeepNeuralNetwork:  0.803170690812    0.78890509725


#Next step is to tune the hyperparameters of our best performing classifiers


#Random Forest Hyperparameter tuning
import numpy as np
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


X, y = X_train,Y_train

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [10,3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3,10,100, None],
              "max_features": [1, 3,7,10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10,100],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



#XGBoost Classifier Hyperparameter tuning

import numpy as np
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


X, y = X_train,Y_train

# build a classifier
clf = XGBClassifier()


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



# use a full grid over all parameters
param_grid = {'nthread':[1,2,3,4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.01,0.05], #so called `eta` value
              'max_depth': [6,8,10,100],
              'min_child_weight': [5,10,11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.3,0.7],
              'n_estimators': [5,10,20,100], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [27,1337]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



import numpy as np
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


X, y = X_train,Y_train

# build a classifier
clf = XGBClassifier()


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



# use a full grid over all parameters
param_grid = {'nthread':[1,2,3,4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.01,0.05,0.1], #so called `eta` value
              'max_depth': [6,8,10,100],
              'min_child_weight': [5,10,11,100],
              'silent': [1,2],
              'subsample': [0.8,0.6],
              'colsample_bytree': [0.3,0.7],
              'n_estimators': [5,10,20], #number of trees, change it to 1000 for better results
              'missing':[-999,0,1],
              'seed': [27,1337,100]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)






#Gradient Boosting Classifier Hyperparameter tuning


import numpy as np
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


X, y = X_train,Y_train

# build a classifier
clf = GradientBoostingClassifier()


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



# use a full grid over all parameters
param_grid = {'max_depth':range(6,16,2),
              'min_samples_split':range(100,999,150),
              'min_samples_leaf':range(20,77,100),
              'max_features':range(7,20,100),
              'subsample':[0.6,0.75,0.8,0.85,0.9]}


# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)





