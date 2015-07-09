import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Read yelp.csv into a DataFrame
yelp = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/hw/optional/yelp.csv')
yelp.head()
yelp.info()
yelp.describe()

# Explore the relationship between each of the vote types (cool/useful/funny) and the number of stars.
# probably a better way to do this
sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=4.5, aspect=0.7)
sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=4.5, aspect=0.7, kind='reg')

fig, axs = plt.subplots(1, 3, sharey=True)
yelp.plot(kind='bar', x='cool', y='stars', ax=axs[0], figsize=(16, 6))
yelp.plot(kind='scatter', x='useful', y='stars', ax=axs[1])
yelp.plot(kind='scatter', x='funny', y='stars', ax=axs[2])

sns.pairplot(yelp)
pd.scatter_matrix(yelp, figsize=(12, 10))

# Use a **correlation matrix** to visualize the correlation between all numerical variables.
# compute correlation matrix
yelp.corr()
# display correlation matrix in Seaborn using a heatmap
sns.heatmap(yelp.corr())

# Define cool/useful/funny as the features, and stars as the response.
feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

feature_cols = ['biz_review_count', 'user_review_count', 'user_mean_rating', 'biz_mean_rating', 'cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

# Fit a linear regression model and interpret the coefficients. 
# Do the coefficients make intuitive sense to you? 
# Explore the Yelp website to see if you detect similar trends.
linreg = LinearRegression()
linreg.fit(X, y)
print linreg.intercept_
print linreg.coef_

y_pred = linreg.predict(X)
metrics.r2_score(y, y_pred)
# Evaluate the model by splitting it into training and testing sets and computing the RMSE. 
# Does the RMSE make intuitive sense to you?

def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# include all features: RMSE = 1.184 (best score with combos of these features)
feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
train_test_rmse(X, y)
# cool and useful only: RMSE = 1.196
# cool and funny only: RMSE = 1.194
# useful and funny only: RMSE = 1.210
# cool only: RMSE = 1.211
# useful only: RMSE = 1.212
# funny only: RMSE = 1.210


# p-values for different yelp coefficients
lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp).fit()
print lm.pvalues

'''
Bonus: Think of some new features you could create from the existing data that might be predictive 
of the response. (This is called "feature engineering".) Figure out how to create those features 
in Pandas, add them to your model, and see if the RMSE improves.

'''
yelp['biz_review_count'] = yelp[['business_id', 'useful']].groupby(['business_id']).transform('count')
yelp['user_review_count'] = yelp[['user_id', 'useful']].groupby(['user_id']).transform('count')
yelp.columns
yelp.head()
yelp.user_review_count.describe()

# all initial features plus biz_review_count: RMSE = 1.178 (best RMSE yet!)
feature_cols = ['cool', 'useful', 'funny', 'biz_review_count', 'user']
X = yelp[feature_cols]
train_test_rmse(X, y)

# user_review_count slightly improves: RMSE = 1.1772
feature_cols = ['cool', 'useful', 'funny', 'biz_review_count', 'user_review_count']
X = yelp[feature_cols]
train_test_rmse(X, y)

# but removing biz_review_count is no good: RMSE = 1.184
feature_cols = ['cool', 'useful', 'funny', 'user_review_count']
X = yelp[feature_cols]
train_test_rmse(X, y)

yelp['user_mean_rating'] = yelp[['user_id', 'stars']].groupby(yelp['user_id']).stars.transform('mean')
yelp['biz_mean_rating'] = yelp[['business_id', 'stars']].groupby(yelp['business_id']).stars.transform('mean')

# user_review_count slightly improves: RMSE = 0.5291 (best yet!)
feature_cols = ['cool', 'useful', 'funny', 'biz_review_count', 'user_review_count', 'user_mean_rating', 'biz_mean_rating']
X = yelp[feature_cols]
train_test_rmse(X, y)

# user_mean_rating is important, error jumps: RMSE = 0.7822
feature_cols = ['cool', 'useful', 'funny', 'biz_review_count', 'user_review_count', 'biz_mean_rating']
X = yelp[feature_cols]
train_test_rmse(X, y)

# interestingly, biz_mean_rating is less important, error jumps but less: RMSE = 0.609
feature_cols = ['cool', 'useful', 'funny', 'biz_review_count', 'user_review_count', 'user_mean_rating']
X = yelp[feature_cols]
train_test_rmse(X, y)

# without review ratings, RSME is almost as low as with them: RMSE = 0.535
feature_cols = ['biz_review_count', 'user_review_count', 'user_mean_rating', 'biz_mean_rating']
X = yelp[feature_cols]
train_test_rmse(X, y)

'''Bonus: Compare your best RMSE on testing set with the RMSE for the "null model", which is the 
model that ignores all features and simply predicts the mean rating in the training set for all 
observations in the testing set.
'''
feature_cols = ['mean_stars_only']
yelp['mean_stars_only'] = yelp.stars.mean()
y_true = yelp.stars
y_pred = yelp.mean_stars_only
type(y_true)
print np.sqrt(metrics.mean_squared_error(y_true, y_pred))
# whoa, RMSE is 1.21, not too much worse than using review tags as features only

'''
Bonus: Instead of treating this as a regression problem, treat it as a classification problem 
and see what testing accuracy you can achieve with KNN.
'''
X, y = yelp.drop('stars', axis = 1), yelp['stars']
X = X[['cool', 'useful', 'funny', 'biz_review_count', 'user_review_count', 'biz_mean_rating']]
X.shape
y.shape
from sklearn.neighbors import KNeighborsClassifier  # import class
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)      
knn.score(X_test, y_test)

from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 100, 2)
param_grid = dict(n_neighbors=k_range) # key has to be exactly the name as scikit learn calls it
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]

grid.best_score_     # shows us the best score
grid.best_params_    # shows us the optimal parameters
grid.best_estimator_ # this is the actual model
# knn is accurate 54.62% of the time, and it's not possible to compare this directly to 
# linear regression, because succcess is measured with different metrics
'''
Bonus: Figure out how to use linear regression for classification, and compare 
its classification accuracy to KNN.
'''

y_pred_round = y_pred.tolist()
y_pred_round = [round(i) for i in y_pred_round]
y_pred_round = np.array(y_pred_round)
np.mean(y_pred_round == y) # whoa! 72.8% accuracy, which is superior to knn