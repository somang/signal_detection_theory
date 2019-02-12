import numpy as np
from scipy.stats import norm
from math import exp,sqrt
Z = norm.ppf

import sdt_metrics
from sdt_metrics import dprime, HI, MI, CR, FA, SDT

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)


#The probability density function for norm is:
# norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi) for a real number x.
# The probability density above is defined in the “standardized” form. 
# To shift and/or scale the distribution use the loc and scale parameters. 
# Specifically, norm.pdf(x, loc, scale) is identically equivalent to
# norm.pdf(y) / scale with y = (x - loc) / scale.


# if you accidentally write scipy.stats.norm(mean=100, std=12) instead of
#  scipy.stats.norm(100, 12) or scipy.stats.norm(loc=100, scale=12), then it'll accept it,
#  but silently discard those extra keyword arguments and give you the default (0,1)

# print(norm.cdf(x, mean, sd))

"""
Here is more info. 
First you are dealing with a frozen distribution
 (frozen in this case means its parameters are set to specific values). 
 
 To create a frozen distribution:

import scipy.stats
scipy.stats.norm(loc=100, scale=12)
#where loc is the mean and scale is the std dev
#if you wish to pull out a random number from your distribution
scipy.stats.norm.rvs(loc=100, scale=12)

#To find the probability that the variable has a value LESS than or equal
#let's say 113, you'd use CDF cumulative Density Function
scipy.stats.norm.cdf(113,100,12)
Output: 0.86066975255037792
#or 86.07% probability

#To find the probability that the variable has a value GREATER than or
#equal to let's say 125, you'd use SF Survival Function 
scipy.stats.norm.sf(125,100,12)
Output: 0.018610425189886332
# or 1.86%

#To find the variate for which the probability is given, let's say the 
#value which needed to provide a 98% probability, you'd use the 
#PPF Percent Point Function
scipy.stats.norm.ppf(.98,100,12)
Output: 124.64498692758187

"""

print(norm.sf(0, 0, 1))
print(norm.sf(0, 1.075, 1))


# Display the probability density function (pdf)
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')

# the distribution object can be called (as a function) to fix the shape, location and scale parameters. 
# This returns a “frozen” RV object holding the given parameters fixed.
#Freeze the distribution and display the frozen pdf:
rv = norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# Check accuracy of cdf and ppf:
vals = norm.ppf([0.001, 0.5, 0.999])
print(np.allclose([0.001, 0.5, 0.999], norm.cdf(vals)))

# Generate random numbers:
r = norm.rvs(size=1000)
# And compare the histogram:
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()



def dPrime(hits, misses, cr, fa):
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5/(hits+misses)
    halfFa = 0.5/(fa+cr)
 
    # Calculate hitrate and avoid d' infinity
    hitRate = hits/(hits+misses)
    if hitRate == 1: hitRate = 1-halfHit
    if hitRate == 0: hitRate = halfHit
 
    # Calculate false alarm rate and avoid d' infinity
    faRate = fa/(fa+cr)
    if faRate == 1: faRate = 1-halfFa
    if faRate == 0: faRate = halfFa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hitRate) - Z(faRate)
    out['beta'] = exp((Z(faRate)**2 - Z(hitRate)**2)/2)
    out['c'] = -(Z(hitRate) + Z(faRate))/2
    out['Ad'] = norm.cdf(out['d']/sqrt(2))
    return out
    # Note the adjustment of rate=0 and rate=1, to prevent infinite values.

if __name__ == "__main__":
    d_prime = dPrime(18, 12, 3, 27)
    print(d_prime) # should be -153.49%

    ### using sdt_metrics library
    hi,mi,cr,fa = 18, 12, 3, 27
    print(dprime(hi,mi,cr,fa))

    sdt_obj = SDT(HI=18, MI=12, CR=3, FA=27)
    #print(sdt_obj)
    print(sdt_obj.c())
    print(sdt_obj.dprime())