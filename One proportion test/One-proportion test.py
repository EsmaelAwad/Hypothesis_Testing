"""
Back in 'Intro to hypothesis testing' we estimated the variance between a sample
statistic and an unknown population proportion.
Also, we measured the std_err using bootstraping.
The std error was then used to compute the statistic the z_score that we used
scipy.stats norm to get the p-value and make a decision!
A bootstrap while easy, Can be computationally expensive.
This time we will estimate it without using bootstraping.

p The unknown population parameter.
p_hat: Sample proportion
p0: Hypothesized population proportion
z_score = (p_hat - p) / std_err
Under the null hypothesis (H0 is true) p0=p. so,
z = (p_hat - p0 ) / std_err
That what we computed in the 'intro to hypothesis testing'.
Now for the std_err estimation:
    SE = np.sqrt( ( p0 * (1-p0) ) / n )

You might wonder why, I used z_score in this situation and not a t_test like
we did before, Well, look at the equation for t_score:
    (x_bar1 - x_bar2)/ np.sqrt( (std_1 **2 /n_1) + (std_2**2 / n_2 ) )

The x_bar was used twice in the numerator and the denominator!
How, It's used first to estimate the x_bar itself and in the equation of std_1.
and 2 offcourse!
This increases the uncertainty of our parameter, Since t-distribution is like
a normal distribution but has a fatter tail to consider outliers and uncertainty
we used it.

let's get back to stack overflow
H0: proportion of stackoverflow under the age of 30 = 0.5
H1: It's not equal to 0.5

"""

import pandas as pd
import numpy as np

stack_overflow = pd.read_feather("stack_overflow.feather")

age_prop = stack_overflow['age_cat'].value_counts(normalize=True)
print(age_prop)

"""

Under 30       0.535604
At least 30    0.464396

"""

p_hat = age_prop["Under 30"]
p_0 = 0.5

numerator = p_hat-p_0
denominator = np.sqrt( p_0 * (1-p_0) / len(stack_overflow) )
z_score = numerator / denominator
print(z_score)
from scipy.stats import norm
#Since this is a two tailed test, we want to account for the whole area that
#adresses the interest of our test

p_value_left_ail_test = norm.cdf(z_score)
p_value_right_tail_test = 1 - norm.cdf(z_score)
p_value_two_tail_test = 2 * (1 - norm.cdf(z_score))
print(p_value_two_tail_test) #0.000709422

#We reject the null hypothesis concluding that indeed, The population under 30 is not equal to 0.5 .

