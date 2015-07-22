##### Part 1 #####

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# 1. read in the yelp dataset
yelp = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/hw/optional/yelp.csv')
yelp.head()
yelp.info()
yelp.describe()

# 2. Perform a linear regression using 
# "stars" as your response and 
# "cool", "useful", and "funny" as predictors

feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

X_train, X_test, y_train, y_test = train_test_split(X, y)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
print linreg.intercept_
print linreg.coef_

y_pred = linreg.predict(X_test)

# 3. Show your MAE, R_Squared and RMSE
# without cv first
metrics.mean_absolute_error(y_test, y_pred) # MAE = 0.9488

y_pred = linreg.predict(X_test)
metrics.r2_score(y_test, y_pred)  # R_Squared = 0.0105

def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
train_test_rmse(X, y) # RMSE = 1.184

# now with cv
from sklearn.cross_validation import cross_val_score
linreg_MAE_scores = cross_val_score(linreg, X, y, scoring='mean_absolute_error', cv=5)
print linreg_MAE_scores
print linreg_MAE_scores.mean() # hmm, MAE from the cross-validator is negative (-0.9488) vs positive above; issue with my function?

linreg_r2_scores = cross_val_score(linreg, X, y, scoring='r2', cv=5)
print linreg_r2_scores
print linreg_r2_scores.mean() # R_squared = 0.0337

linreg_MSE_scores = cross_val_score(linreg, X, y, scoring='mean_squared_error', cv=5)
print linreg_MSE_scores
MSE = linreg_MSE_scores.mean()
import cmath
RMSE = cmath.sqrt(MSE) 
print RMSE # RMSE = 1.1937

# 4. Use statsmodels to show your pvalues
# for each of the three predictors
# Using a .05 confidence level, 
# Should we eliminate any of the three?
lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp).fit()
print lm.pvalues
'''
pvalues
cool: 2.989e-90
useful: 1.206e-39
funny: 1.851e-43
no need to eliminate any of the 3
'''

# 5. Create a new column called "good_rating"
# this could column should be True iff stars is 4 or 5
# and False iff stars is below 4
yelp['good_rating'] = (np.where(yelp['stars'] > 3, True, False))
yelp.good_rating.mean()

# 6. Perform a Logistic Regression using 
# "good_rating" as your response and the same
# three predictors
feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.good_rating

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_test_preds = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_test_preds)


# 7. Show your Accuracy, Sensitivity, Specificity
# and Confusion Matrix
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(logreg, X, y, scoring='accuracy', cv=5)
print scores
print scores.mean() # cv accuracy = 0.6958

y_preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, y_preds)
cm = metrics.confusion_matrix(y_test, y_preds)
'''
confusion matrix:
[[  52  727]
 [  28 1693]]
 '''
sensitivity = (float(cm[0,0]) / (cm[0,0] + cm[1,0]))
specificity = (float(cm[1,1]) / (cm[1,1] + cm[0,1]))
# sensitivity = 65%
# specificity = 69.9%%

# 8. Perform one NEW operation of your 
# choosing to try to boost your metrics!

yelp['user_mean_rating'] = yelp[['user_id', 'stars']].groupby(yelp['user_id']).stars.transform('mean')
yelp['biz_mean_rating'] = yelp[['business_id', 'stars']].groupby(yelp['business_id']).stars.transform('mean')
yelp['biz_review_count'] = yelp[['business_id', 'useful']].groupby(['business_id']).transform('count')
yelp['user_review_count'] = yelp[['user_id', 'useful']].groupby(['user_id']).transform('count')

feature_cols = ['cool', 'useful', 'funny', 'biz_review_count', 'user_review_count', 'user_mean_rating', 'biz_mean_rating']
X = yelp[feature_cols]
y= yelp.stars

linreg_MAE_scores = cross_val_score(linreg, X, y, scoring='mean_absolute_error', cv=5)
print linreg_MAE_scores
print linreg_MAE_scores.mean() # new MAE = -0.3885

linreg_r2_scores = cross_val_score(linreg, X, y, scoring='r2', cv=5)
print linreg_r2_scores
print linreg_r2_scores.mean() # R_squared = 0.801

linreg_MSE_scores = cross_val_score(linreg, X, y, scoring='mean_squared_error', cv=5)
print linreg_MSE_scores
MSE = linreg_MSE_scores.mean()
import cmath
RMSE = cmath.sqrt(MSE) 
print RMSE # RMSE = 0.5419, woohoo!

##### Part 2 ######

# 1. Read in the titanic data set.
data = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/data/titanic.csv')
# so that 'Sex' can be used in a model:
data.Sex = data.Sex.replace(to_replace='male', value=1)
data.Sex = data.Sex.replace(to_replace='female', value=0)
data.head()

# 4. Create a new column called "wife" that is True
# if the name of the person contains Mrs.
# AND their SibSp is at least 1
data['wife'] = False
data['wife'][(data['Name'].str.contains('Mrs.')) & (data['SibSp'] >= 1)] = True
data.head()

# 5. What is the average age of a male and
# the average age of a female on board?
data.groupby('Sex')['Age'].mean() # female = 27.9 yrs, male = 30.7 yrs

# 5. Fill in missing MALE age values with the
# average age of the remaining MALE ages
avg_age_male = data.Age[data.Sex == 1].mean()
data.Age[data.Sex == 1] = data.Age[data.Sex == 1].fillna(value=avg_age_male)

# 6. Fill in missing FEMALE age values with the
# average age of the remaining FEMALE ages
avg_age_female = data.Age[data.Sex == 0].mean()
data.Age[data.Sex == 0] = data.Age[data.Sex == 0].fillna(value=avg_age_female)

# 7. Perform a Logistic Regression using
# Survived as your response and age, wife
# as predictors

X = data[[ 'Age', 'wife']]
y = data.Survived
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 8. Show Accuracy, Sensitivity, Specificity and 
# Confusion matrix

scores = cross_val_score(logreg, X, y, scoring='accuracy', cv=10)
print scores
print scores.mean() # accuracy = 0.667

y_preds = logreg.predict(X_test)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_preds)
'''
confusion matrix
[[127,   8],
[ 69,  19]]
'''
sensitivity = (float(cm[0,0]) / (cm[0,0] + cm[1,0])) # sensitivity = 64.8%
specificity = (float(cm[1,1]) / (cm[1,1] + cm[0,1])) # specificity = 70.37% 

# 9. now use ANY of your variables as predictors
# Still using survived as a response to boost metrics!
pclass_dummies = pd.get_dummies(data.Pclass, prefix='Pclass')
data = data.merge(pclass_dummies, right_index=True, left_index=True)
data.head()

X = data[['Sex', 'Age', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Parch', 'wife']]
y = data.Survived

# 10. Show Accuracy, Sensitivity, Specificity
X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg.fit(X_train, y_train)

scores = cross_val_score(logreg, X, y, scoring='accuracy', cv=10)
print scores
print scores.mean() # accuracy = 0.7969

y_preds = logreg.predict(X_test)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_preds)

sensitivity = (float(cm[0,0]) / (cm[0,0] + cm[1,0])) # sensitivity = 86.36%
specificity = (float(cm[1,1]) / (cm[1,1] + cm[0,1])) # specificity = 72.53%