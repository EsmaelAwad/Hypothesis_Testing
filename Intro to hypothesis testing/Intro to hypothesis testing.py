#Reading the data in the same directory as the work file
import pandas as pd
stack_overflow = pd.read_feather("stack_overflow.feather")
#Checking the shape..
print(stack_overflow.shape)
#We have 2261 rows, 63 columns

"""
Our hypothesis is that the mean compensation of the population of data scientis
is 110,000$

"""
mean_comp_sample = stack_overflow['converted_comp'].mean()

print(mean_comp_sample) #119,574.71$

"""
The result is different than our hypothesis, But is it meaningly different?
AKA: Is is due to random chance?

To answer this we need to generate a bootstrap distribution of sample means.

"""

import numpy as np

np.random.seed(42)

sample_means = []

for i in range(1000):
    
    """
    Here we are appending the mean of the results of sampling bootstrap to
    sample_means.
    df.sample(...) returns a dataframe that you can subset using pandas syntax
    df['col'].
    
    """
    
    sample_means.append(
        np.mean(
            stack_overflow.sample(frac=1,replace=True)['converted_comp']
            )
        )

import matplotlib.pyplot as plt

print(np.mean(sample_means))
plt.hist(sample_means)
plt.title("Sample means Distributions")
plt.show()
plt.clf()

#As you can see our data is roughly normally distributed!

std_err = np.std(sample_means,ddof=1) #The std of the bootstrap is the std_err
print(std_err) # 5486.361424047341
"""
Since our values are arbitary units, We need to standardize it so we can get
The z score and test our hypothesis, To do so, we use the followwing formula

Our hypothesis mean = 110,000
The sample mean = 119574.71738168952
The std error = 5486.361424047341
"""

Hypo_mean = 110000

z_score = (mean_comp_sample - Hypo_mean)/std_err

print(z_score) #1.7451853134797968

"""
Is that a big or small number?
This is the heart of hypothesis testing, Detirmining wether a sample statistic
is far or close to the expected (population) summary statistic(mean in our case)
This is what we will detirmine now.

previous research states that : 35% of data scientists started programming as
                                children
in our data we have a column called 'age first cut' which has two values:
    'adult': for the programmer who started programming after 14 years old
    'child': started before 14
        
H0: The proportion of programmers whos started coding early is 35%
H1: The proportion of programmers whos started coding early is > 35%

Since the test in favor is a 'greater than' test, we'll perform a right-tail test

large p-values are support to H0 while small p-value are evidence against H0

why? Because if you think about the bell-curve, the large p-value means that 
the observed values lie in the normal distribution of the data, meaning that
the observed sample that shows unlikley kind of data is just due to chance!

while low p-values mean that the values are on the tails of the distribution
Which means that the data provided a good evidince that the data that we thought
was TRUE turned out to be not. Therefore, We reject the H0

"""

prop_child_sample = (stack_overflow["age_first_code_cut"]=="child").mean()

print(prop_child_sample)#0.39141972578505085

prob_child_Hypothesis = 0.35

sample_means = []

for i in range(1000):
    
    """
    Here we are appending the mean of the results of sampling bootstrap to
    sample_means.
    df.sample(...) returns a dataframe that you can subset using pandas syntax
    df['col'].
    
    """
    
    sample_means.append(
        np.mean(
            (stack_overflow['age_first_code_cut']=="child").sample(frac=1,replace=True)
            )
        )

std_error = np.std(sample_means,ddof=1)

z_score = (prop_child_sample - prob_child_Hypothesis)/std_error

print(z_score) #4.184454014421004

from scipy.stats import norm

"""
Since this is a right-tail test, we want all values that are greater(to the right)
of our z_score.
The CDF gives us the probability of obtaining a value equal to or LESS than 
our z-score. so, we can simply get that and substract it from 1.

NOTE: Normal standard distribution has a mean of 0 and a std of 1

"""

p_value = 1 - norm.cdf(z_score,loc=0,scale=1)
print(p_value)#1.4292615151179078e-05 Which is a very small number, so, we reject the H0

"""
The p-value = 1.4292615151179078e-05 which is a very small number, hence, strong
evidence against H0, We accept H1 (Reject the H0) Indeed, more than 35% of programmers
started coding before the age of 14.

"""



