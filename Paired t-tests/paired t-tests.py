import pandas as pd
import numpy as np

republic_votes_08_12 = pd.read_feather("repub_votes_potus_08_12.feather")

"""
we have 2 columns called:
    1- repub_percent_08
    2- repub_percent_12
we want to answer this question, was the 08 votes percentegas lower than 2012?

Our hypothesis:
    H0: M(2008) - M(2012) = 0
    H1: M(2012) - M(2008) > 0 --right tailed test.

since both columns refer to the same county, then these are paired (dependant)
variables. Which means: voting patterns may occur! We want to capture that in
our model!

for paired analyses, rather than compairing the two variables seperately,
we can consider a diff variable as the difference between them, it's mean then
would be our sample statistic!

      state       county  repub_percent_08  repub_percent_12
0      Alabama         Hale         38.957877         37.139882
1     Arkansas       Nevada         56.726272         58.983452
2   California         Lake         38.896719         39.331367
3   California      Ventura         42.923190         45.250693
4     Colorado      Lincoln         74.522569         73.764757
..         ...          ...               ...               ...
95   Wisconsin      Burnett         48.342541         52.437478
96   Wisconsin    La Crosse         37.490904         40.577038
97   Wisconsin    Lafayette         38.104967         41.675050
98     Wyoming       Weston         76.684241         83.983328
99      Alaska  District 34         77.063259         40.789626

"""
republic_votes_08_12['diff'] = republic_votes_08_12['repub_percent_12']\
    -republic_votes_08_12['repub_percent_08']

x_bar = republic_votes_08_12['diff'].mean() #2.8771090412429454

import matplotlib.pyplot as plt

plt.figure(figsize=(25,15))

plt.title('Difference Distribution')

republic_votes_08_12['diff'].hist(bins=20,grid=False)

"""
Most values are in between -3 and 10 while having atleast one outlier > -30 !

You can see the graph in the folder of this test as 'difference distribution plot'

We have a new hypothesis:
    H0: Diff = 0
    H1: Diff > 0 (2012 is greater than 2008)

t = ( X_bar(diff) - population_mean(diff) ) / std_err(diff)

if the H0 is assumed True then the population mean(diff) = 0

degrees of freedom = len(data[diff]) - 1

"""
deg_free = len(republic_votes_08_12) - 1 # 99

n_diff = len(republic_votes_08_12['diff']) # 100

s_diff = republic_votes_08_12['diff'].std()

print(s_diff)#5.136737887231852

t_statistic = (x_bar - 0) / np.sqrt(s_diff**2 / n_diff )

from scipy.stats import t

p_value = 1 - t.cdf(t_statistic, df=deg_free)

print(p_value) # 9.572537285063021e-08
"""
again, a small number, saying that our H0 is rejected, indeed there's a 
Significant difference towards 2012 (2012 was higher)

However, This could have been also achieved through bootsraping, but this is 
the academic way of doing so.

That was a lot of computing for a single test, However, There's a simpler
way to do that using the pingouin module instead of scipy testing functions.
Because it's way easier to interpret it's results.

"""
import pingouin as pn

t_test= pn.ttest(x=republic_votes_08_12['diff'],
                 y = 0, #because we calculated the differences between x and y
                 alternative = "greater" #Could have been "less" or "two-sided"
                 )

pd.set_option('display.expand_frame_repr', False)

print(t_test)

"""
               T  dof alternative         p-val        CI95%   cohen-d       BF10  power
T-test  5.601043   99     greater  9.572537e-08  [2.02, inf]  0.560104  1.323e+05    1.0               

As you can see these numbers are the same as we calculated them! p-val is 
9.572537e-08 stating the significance of our sample data.

The T statistic is as calculated, dof is the degrees of freedom which is 100-1
The CI95% is the range of the distribution to the right of the mean.

However, we could have made this exact test without calculating the diff column.

"""

t_test_no_diff_pair_wise = pn.ttest(x=republic_votes_08_12['repub_percent_12'],
                                    y=republic_votes_08_12['repub_percent_08'],
                                    alternative="greater",
                                    paired=True)

print(t_test_no_diff_pair_wise)

"""
               T  dof alternative         p-val        CI95%   cohen-d       BF10     power
T-test  5.601043   99     greater  9.572537e-08  [2.02, inf]  0.217364  1.323e+05  0.696338

Again This is exactly the same!

"""

t_test_no_diff_pair_wise_type1_err = pn.ttest(x=republic_votes_08_12['repub_percent_12'],
                                    y=republic_votes_08_12['repub_percent_08'],
                                    alternative="greater",
                                    paired=False)

print(t_test_no_diff_pair_wise_type1_err)

"""
               T  dof alternative     p-val         CI95%   cohen-d   BF10     power
T-test  1.536997  198     greater  0.062945  [-0.22, inf]  0.217364  0.927  0.454972

The test was calculated as if the two samples were independant, Resulting in a
p-value > 0.05 which would have mislead us to failing to reject the H0 (accept it)
and saying that the data we collected is just due to random chance!
This is known as type1 error which is a false negative type of error.
search on the internet for Type 1 and type 2 errors to know more.

"""

"""

I know you might say (I am confused! what if by mistake I conducted a left-tail
                      for a dataset that is meant to be tested right!)
Having been there myself, I say once you define your hypothesis test correctly
you can never get it wrong, Practice really makes perfect!
 You must follow theses steps:
    1- Identify population parameter that is hypothesized about.
    2- Specify the null and alternative hypotheses.
    3- Determine (standardized) test statistic and corresponding null distribution.
    4- Conduct hypothesis test in Python.
    5- Measure evidence against the null hpothesis. (p_value).
    6- Make a decision comparing evidence to significance level.

However, if you've done it wrong, The p-value would still give you a number, But it
will be completly wrong, Throwing you to the Type 2 error!

To test that:
    
"""

t_test_that_is_right_made_left = pn.ttest(x=republic_votes_08_12['repub_percent_12'],
                                    y=republic_votes_08_12['repub_percent_08'],
                                    alternative="less", #foucs here ----------
                                    paired=True)

print(t_test_that_is_right_made_left)

"""
               T  dof alternative  p-val         CI95%   cohen-d       BF10     power
T-test  5.601043   99        less    1.0  [-inf, 3.73]  0.217364  7.558e-06  0.000071

Notice how the T statistic is the same as all the other test, Because the distribution
is symmetrical, we get the same result always.
but look at the p-value it freaking huge! There's no way you could have rejected
the null hypothesis!

That's why you need to be careful and do not ignore any of the steps above.

"""
