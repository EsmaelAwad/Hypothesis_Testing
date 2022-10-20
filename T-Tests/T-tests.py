"""
Back again to our stackoverflow data, We have two columns:
    converted comp is a numerical column of compensations given to data scientists
    age_first_code is a categorical with two values('child','adult') for those
    who started their career as kids or as adults.
You can already initiate a test: are those who started programming as childs
                                better compensated than those who started when
                                they were adults?
H0: Mean compensation for 'child' = mean compensation for 'adult' Default-Case
H1: Mean compensation for 'child' > mean compensation for 'adult' Challenger-Case

or
H0: M('child') = M('adult') or M('child') - M('adult') = 0
H1: M('child') > M('adult') or M('child') - M('adult') > 0

"""
import pandas as pd
import numpy as np
stack_overflow = pd.read_feather("stack_overflow.feather")

comp_adult_child_group = stack_overflow.groupby("age_first_code_cut")['converted_comp'].mean()
print(comp_adult_child_group)#adult:111313.311047 , child:132419.570621

"""
Clearly the child attribute shows higher values, But is that really the case?
is this result statistically significant? or is it just due to chance?
Let's find out!

"""

"""
Remember how to standerdize the z-score using the z formula? (Sample mean- pop mean)/std_err

Now, in the T-test we are doing a test not for the means, but, for the difference
in means! and the formula is as follows:

    t = (X_bar Child - X_bar Adult) - (Pop_mean Child - Pop_mean Adult)/Std_err

Calculating the std_err using Bootstraping is a good option, However, We can 
Estimate it using a formula:
    
    np.sqrt((std_child**2 / n_child) + (std_adult**2 / n_adult))

If we assumed the H0 to be True, Then (Pop_mean Child - Pop_mean Adult) = 0
Then the formula can be:
    
    (X_bar Child - X_bar Adult) - 0 /Std_err
    
"""
#This process can be afunction to use later but for demonstration, let's code!

x_bar = comp_adult_child_group #adult:111313.311047 , child:132419.570621
s = comp_adult_child_group = stack_overflow.groupby("age_first_code_cut")['converted_comp'].std()
n = comp_adult_child_group = stack_overflow.groupby("age_first_code_cut")['converted_comp'].count()

x_bar_adult = x_bar['adult']
x_bar_child = x_bar['child']

s_adult = s['adult']
s_child = s['child']

n_adult = n['adult']
n_child = n['child']

numerator = x_bar_child - x_bar_adult
denominator = np.sqrt( (s_child**2 / n_child) + (s_adult**2 / n_adult) )
t_stat = numerator/denominator

print(t_stat) #1.8699313316221844

from scipy.stats import t

p_value = 1 - t.cdf(t_stat, df=len(stack_overflow) - 2)
print(p_value)

# 0.030811302165157595 which is lower than 0.05 so we reject the H0, indeed,\
    #people who started coding as children, Earned more.

"""
What is Degrees of freedom (df)?
Suppose we have 5 independent data points, four of them are [2,6,8,5]
and we know that the mean of the whole set of the 5 numbers is: 5
which means that the 5th number is no longer 'independant', it must be 4 to
get to the mean of 5.
This means this dataset has a degree of freedom that is equal to n-1 or 5-1=4

What does that has to do with the t distribution?
because the t- distribution is like a normal distribution but with fatter tales
because it measures and estimates population sample statistics for small sets
of samples, which means, the confidence intervals are wider than the normal distribution.

in our data we caluclated the x_bar_child and adult so we have a degree of freedom
that is egual to : n-2
as the degrees of freedom increases the closer we get to the normal distribution
hence, increasing the confidence of our sample statistic!

"""


"""

Hypothesis testing work flow:
    1- Identify population parameter that is hypothesized about. --done
    2- Specify the null and alternative hypotheses. --done
    3- Determine (standardized) test statistic and corresponding null distribution. --done
    4- Conduct hypothesis test in Python. --done
    5- Measure evidence against the null hypothesis. (p_value) --done
    6- Make a decision comparing evidence to significance level. --done
        
"""