#Importing packages
import pandas as pd
import numpy as np

stack_overflow = pd.read_feather("stack_overflow.feather")
print(stack_overflow.columns)
"""
The stack overflow contains a hobbyist variable:
    Yes: Means the user describes himself as a hobbiest
    No: He is a professional
Also, We have the age_cat ( under 30 or not)

H0: Proportion of hobbyist users is the same for those under thirty as those
    at least thirty

H1: Proportion of hobbyist users is different for those under thirty to those
    at least thirty

"""

p_hats = stack_overflow.groupby("age_cat")['hobbyist'].value_counts(normalize=True)

print(p_hats)
"""
age_cat      hobbyist
At least 30  Yes         0.773333
             No          0.226667
Under 30     Yes         0.843105
             No          0.156895

"""
n = stack_overflow.groupby("age_cat")['hobbyist'].count()

print(n) 

"""
age_cat
At least 30    1050
Under 30       1211

"""

#Since we're only interested in testing those who are hobbyists we'll choose only
#values where hobbyist is equal to Yes

p_hat_at_least_30 = p_hats[("At least 30","Yes")]
p_hat_under_30 = p_hats[("Under 30","Yes")]

print(p_hat_at_least_30,p_hat_under_30) #0.7733333333333333, 0.8431048720066061

n_atleast_30 = n["At least 30"]
n_under_30 = n["Under 30"]
print(n_atleast_30,n_under_30) # 1050 , 1211

p_hat = (n_atleast_30 * p_hat_at_least_30 + n_under_30 * p_hat_under_30)/(
    n_atleast_30 + n_under_30)

std_error = np.sqrt(p_hat * (1-p_hat) / n_atleast_30 +
                    p_hat * (1-p_hat) / n_under_30)

z_score = (p_hat_at_least_30 - p_hat_under_30) / std_error

print(z_score) #-4.22

#we can avoid this much work by providing numpy arrays to statsmodel
#n_hobbyists at least 30: 812 n_hobbyist under 30 : 1021
"""
stack_overflow.groupby("age_cat")['hobbyist'].value_counts()

age_cat      hobbyist
At least 30  Yes          812
             No           238
Under 30     Yes         1021
             No           190
             
"""
print(812+238)
n_hobbyists = np.array([812,1021])
n_rows = np.array([812+238,1021+190])

from statsmodels.stats.proportion import proportions_ztest
z_score,p_value = proportions_ztest(count = n_hobbyists, nobs = n_rows,
                                    alternative="two-sided")

print(z_score,p_value) #-4.223691463320559 2.403330142685068e-05
#Just as we calculated it, The p_value is a small number, Suggesting that we 
#Should reject The null hypothesis, There is a statistically significance difference
#between the groups we tested!

#Note: We input the 812 for the number of hobbyists who are at least 30, and for
#The (812+238) this is the total sample size of at least 30
#Can we make the process of np.array([812,1021]) with a loop?

n_test = stack_overflow.groupby("age_cat")['hobbyist'].value_counts()

print(n_test.iloc[n_test.index.get_level_values("hobbyist") == "Yes"]) #1
print([x for x in n_test.iloc[n_test.index.get_level_values("hobbyist") == "Yes"]]) #2
print([x for x in stack_overflow["age_cat"].value_counts().sort_index()]) #3
"""
1:
    age_cat      hobbyist
    At least 30  Yes          812
    Under 30     Yes         1021
2:
    [812, 1021]

3:
    [1050, 1211]
    
So, No matter how your data changes you can get the entries for the arrays automatically.

Why don't you make this whole thing a function that you can reuse?

"""
