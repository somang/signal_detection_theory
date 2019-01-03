from scipy.stats import norm
from math import exp,sqrt
Z = norm.ppf
 
def dPrime(hits, misses, fas, crs):
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5/(hits+misses)
    halfFa = 0.5/(fas+crs)
 
    # Calculate hitrate and avoid d' infinity
    hitRate = hits/(hits+misses)
    if hitRate == 1: hitRate = 1-halfHit
    if hitRate == 0: hitRate = halfHit
 
    # Calculate false alarm rate and avoid d' infinity
    faRate = fas/(fas+crs)
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
    d_prime = dPrime(27, 3, 12, 18)
    print(d_prime) # should be -153.49%