

def ratio_enumerator(L):
    _min, _max, _len = min(L), max(L), len(L)
    _rng = _max - _min
    
    for f in L:
        yield (f-_min)/_rng, f
