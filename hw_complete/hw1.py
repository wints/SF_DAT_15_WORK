'''
Move this code into your OWN SF_DAT_15_WORK repo

Please complete each question using 100% python code

If you have any questions, ask a peer or one of the instructors!

When you are done, add, commit, and push up to your repo

This is due 7/1/2015
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# pd.set_option('max_colwidth', 50)
# set this if you need to

killings = pd.read_csv('hw/data/police-killings.csv')
killings.head()
killings.describe

# 1. Make the following changed to column names:
# lawenforcementagency -> agency
# raceethnicity        -> race
killings.rename(columns={'lawenforcementagency':'agency',
                         'raceethnicity': 'race'}, inplace=True)
killings.columns
killings.describe()
killings.shape

# 2. Show the count of missing values in each column

# count does this pretty nicely; easy to eyeball in this case and ID streetaddress as
# the missing data culprit.
killings.count()

# but this is even better; thx stackoverflow
len(killings.index)-killings.count()


# 3. replace each null value in the dataframe with the string "Unknown"
killings.fillna(value='Unknown', inplace=True)

# 4. How many killings were there so far in 2015?
killings.year.value_counts() # they're all in 2015

# 5. Of all killings, how many were male and how many female?
killings.gender.value_counts()

# 6. How many killings were of unarmed people?
killings.armed.describe()
killings.armed.unique()
killings[killings.armed=='No'].count() #102, if only filtering on 'No'; mightbe valid to include 'Disputed'
killings.armed.value_counts() # or this, which I prefer because it gives some context

# 7. What percentage of all killings were unarmed?
killings[killings.armed=='No'].count() / killings.count() * 100 #21.84%
killings.armed.value_counts() / len(killings.index) * 100 # i like this better

# 8. What are the 5 states with the most killings?
killings.state.value_counts().head() # CA, TX, FL, AZ, OK

# 9. Show a value counts of deaths for each race
killings.race.value_counts()

# 10. Display a histogram of ages of all killings
killings.age.hist(bins=100)

# 11. Show 6 histograms of ages by race

# having a hard time adding labels as I'd like to, but I guess this answers the question
killings.age.hist(by=killings.race, sharex=True, sharey=True, figsize=(10,10))

# 12. What is the average age of death by race?
killings.groupby('race').age.mean()

# just messing around with groupby
killings.groupby(['race', 'gender'])['age', 'county_income'].mean()
# 13. Show a bar chart with counts of deaths every month

killings.month.value_counts().plot(kind='bar')

# would rather sort by month chronologically, not by quantities
killings_month = killings.month.value_counts()
mi = [2,3,0,1,4,5]
pd.DataFrame(data=dict(killings_month), index=mi) # not quite what I was hoping for


###################
### Less Morbid ###
###################

majors = pd.read_csv('hw/data/college-majors.csv')
majors.head()

# 1. Delete the columns (employed_full_time_year_round, major_code)
del majors['Employed_full_time_year_round']
del majors['Major_code']
majors.describe()

# 2. Show the cout of missing values in each column
len(majors.index)-majors.count() # 0? this seems suspicious.

# 3. What are the top 10 highest paying majors?

majors.groupby('Major').Median.mean().order().tail(10)# by avg media salary

# 4. Plot the data from the last question in a bar chart, include proper title, and labels!
salary_bar = majors.groupby('Major').Median.mean().order().tail(10).plot(kind='bar', title='Top 10 Avg Median Salaries by Major (USD)')

# would prefer some better labeling; no error, but also not printing 
salary_bar.set_xlabel("Major")
salary_bar.set_xlabel("Salary")
salary_bar

# 5. What is the average median salary for each major category?
mcat_salaries_all = majors.groupby('Major_category').Median.mean().order()
mcat_salaries_all

# 6. Show only the top 5 paying major categories
majors.groupby('Major_category').Median.mean().order().tail(5)

# 7. Plot a histogram of the distribution of median salaries
mcat_salaries_all.plot(kind='bar', title='Avg Median Salary (USD) by Major Category')

# 8. Plot a histogram of the distribution of median salaries by major category
mcat_salaries_all.hist(bin=10)

# 9. What are the top 10 most UNemployed majors?
# in absolute terms:
majors.groupby('Major')['Unemployed'].sum().order().tail(10)

# What are the unemployment rates?
major_un = majors.groupby('Major')['Total', 'Unemployed'].sum()

major_un_rate = major_un.Unemployed / major_un.Total * 100 
major_un_rate.order().tail(10)


# 10. What are the top 10 most UNemployed majors CATEGORIES? Use the mean for each category
majors.groupby('Major_category')['Unemployed'].sum().order().tail(10)

# What are the unemployment rates?
major_cat_un = majors.groupby('Major_category')['Total', 'Unemployed'].sum()

major_cat_un_rate = major_cat_un.Unemployed / major_cat_un.Total * 100
major_cat_un_rate.order().tail(10)

# 11. the total and employed column refer to the people that were surveyed.
# Create a new column showing the emlpoyment rate of the people surveyed for each major
# call it "sample_employment_rate"
# Example the first row has total: 128148 and employed: 90245. it's 
# sample_employment_rate should be 90245.0 / 128148.0 = .7042
majors['sample_employment_rate'] = majors['Employed'] / majors['Total']
majors.sample_employment_rate

# 12. Create a "sample_unemployment_rate" colun
# this column should be 1 - "sample_employment_rate"
majors['sample_unemployment_rate'] = 1 - majors['sample_employment_rate']
majors.sample_unemployment_rate
