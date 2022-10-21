#Importing Neccesary pacakages
import pingouin as pn
import pandas as pd
#Loading the data in directory
stack_overflow = pd.read_feather("stack_overflow.feather")

print(stack_overflow['job_sat'].value_counts())

"""
Very satisfied           879
Slightly satisfied       680
Slightly dissatisfied    342
Neither                  201
Very dissatisfied        159
Name: job_sat, dtype: int64

ANOVA tests are tests on variance between categorical variables, it's used to
test for statistical significance between two or more pairs of categories.
In english: You want to see which pair of categories has a population parameter
            More varied than the one observed. for example:
                You want to check wether the difference between very satisfied
                and slightly satisfied is really the difference in real world,
                Not just random chance.
Just like any other test we performed earlier, we have:
    a question: Is the mean annual compensation different for different levels
    of job satisfaction?
    A hypothesis: H0: There's no difference across categories
                  H1: There's a difference
So, we want to see what the compensation looks like in our sample data across
different categories.

Still confused? Check out this article:https://www.simplypsychology.org/anova.html

"""
import seaborn as sns
import matplotlib.pyplot as plt

#Setting the size of the chart
plt.figure(figsize=(30,15))

_= sns.boxplot(x="converted_comp",
               y="job_sat",
               data = stack_overflow)
#You can check out the plot in the same directory in the repository with the name as the title
plt.title("Distribution of compensations across multiple job satisfaction")
plt.show()
plt.clf()

"""
Very satisfied seems to have higher compensation rates, But to really state the
test, We have to perform hypothesis tests.

"""
#This number is larger than the 0.05 we used to set for alpha, but it'll help us
#understand more about the test p-value later on.
alpha = 0.2

#To show all columns (Expand the dataframe returned by pingouin methods)
pd.set_option('display.expand_frame_repr', False)

anova_test = pn.anova(data=stack_overflow,
               dv="converted_comp",
               between="job_sat")
#print(anova_test)

"""
    Source  ddof1  ddof2         F     p-unc       np2
0  job_sat      4   2256  4.480485  0.001315  0.007882

The p-unc is indeed lower than our alpha, Which means we have two pairs or more
that has statistically significance difference (the variances observered are not due to chance)

However, We still do not know which two categories are statistically significant.
We need to perform pairwise tests!

for example Mean(very_satisfied) != Mean(slightly satisfied),
            Mean(very_satisfied) != Mean(very dissatisfied) and so on...

You see where this is going? We perform a test for each possible pair on wether
they are different or not, Without getting into combinatorics mathematics, let's
do that with pingouin!

"""            
pairwise_anova_no_adjusting = pn.pairwise_tests(data = stack_overflow,
                                   dv = "converted_comp",
                                   between = "job_sat",
                                   padjust = "none")

#print(pairwise_anova_no_adjusting)

"""
  Contrast                   A                      B  Paired  Parametric         T          dof alternative     p-unc     BF10    hedges
0  job_sat  Slightly satisfied         Very satisfied   False        True -4.009935  1478.622799   two-sided  0.000064  158.564 -0.192931
1  job_sat  Slightly satisfied                Neither   False        True -0.700752   258.204546   two-sided  0.484088    0.114 -0.068513
2  job_sat  Slightly satisfied      Very dissatisfied   False        True -1.243665   187.153329   two-sided  0.215179    0.208 -0.145624
3  job_sat  Slightly satisfied  Slightly dissatisfied   False        True -0.038264   569.926329   two-sided  0.969491    0.074 -0.002719
4  job_sat      Very satisfied                Neither   False        True  1.662901   328.326639   two-sided  0.097286    0.337  0.120115
5  job_sat      Very satisfied      Very dissatisfied   False        True  0.747379   221.666205   two-sided  0.455627    0.126  0.063479
6  job_sat      Very satisfied  Slightly dissatisfied   False        True  3.076222   821.303063   two-sided  0.002166     7.43  0.173247
7  job_sat             Neither      Very dissatisfied   False        True -0.545948   321.165726   two-sided  0.585481    0.135 -0.058537
8  job_sat             Neither  Slightly dissatisfied   False        True  0.602209   367.730081   two-sided  0.547406    0.118  0.055707
9  job_sat   Very dissatisfied  Slightly dissatisfied   False        True  1.129951   247.570187   two-sided  0.259590    0.197  0.119131

A and B are being compaired at each row.
Look at the p-unc column:
    Three of these are less than our alpha (0.2) indexes (0,4,6)

In this case, We have 5 groups, Resulting in 10 pairs right (from 0 to 9)

as the number of groups increases the number of hypothesis tests increase as well
at a notation of O(n**2).
The more tests we run, The higher the percentage that atleast one will result in
a false positive significance result.
When we set alpha to 0.2 if we run 1 test the chance of a false positive result is 0.2
with 5 groups the probability of getting a false positive result is: 0.7.

We want to eliminate this type of error. But, How?

applying the adjustment to increase the p-values reducing the chances of getting a false positive!

a common adjustment we'll use is the bonf adjustment.

"""

anova_test_adjusted = pn.pairwise_tests(data=stack_overflow,
                                        dv = 'converted_comp',
                                        between= "job_sat",
                                        padjust = "bonf")
print(anova_test_adjusted)

"""

  Contrast                   A                      B  Paired  Parametric         T          dof alternative     p-unc    p-corr p-adjust     BF10    hedges
0  job_sat  Slightly satisfied         Very satisfied   False        True -4.009935  1478.622799   two-sided  0.000064  0.000638     bonf  158.564 -0.192931
1  job_sat  Slightly satisfied                Neither   False        True -0.700752   258.204546   two-sided  0.484088  1.000000     bonf    0.114 -0.068513
2  job_sat  Slightly satisfied      Very dissatisfied   False        True -1.243665   187.153329   two-sided  0.215179  1.000000     bonf    0.208 -0.145624
3  job_sat  Slightly satisfied  Slightly dissatisfied   False        True -0.038264   569.926329   two-sided  0.969491  1.000000     bonf    0.074 -0.002719
4  job_sat      Very satisfied                Neither   False        True  1.662901   328.326639   two-sided  0.097286  0.972864     bonf    0.337  0.120115
5  job_sat      Very satisfied      Very dissatisfied   False        True  0.747379   221.666205   two-sided  0.455627  1.000000     bonf    0.126  0.063479
6  job_sat      Very satisfied  Slightly dissatisfied   False        True  3.076222   821.303063   two-sided  0.002166  0.021659     bonf     7.43  0.173247
7  job_sat             Neither      Very dissatisfied   False        True -0.545948   321.165726   two-sided  0.585481  1.000000     bonf    0.135 -0.058537
8  job_sat             Neither  Slightly dissatisfied   False        True  0.602209   367.730081   two-sided  0.547406  1.000000     bonf    0.118  0.055707
9  job_sat   Very dissatisfied  Slightly dissatisfied   False        True  1.129951   247.570187   two-sided  0.259590  1.000000     bonf    0.197  0.119131

Now look at the p_corr. can you detect which pair was a false positive one?
Indeed, It's the 4th index (5th pair) leaving us with only indexes with statistically significance variances (0,6).

There are other methods:
    'none' [default]
    'bonf' one_step bonferrani correction
    'sidak' one-step sidak correction
    'holm' step-down method using bonferroni adjustments
    'fdr_bh' benjamini FDR correction
    'fdr_by' Benjamini Yekutieli FDR correction

Feel free to try them as you like.

"""

