""" A class to report variables"""

import os 
import utils 
from collections import defaultdict


# class Reporter(object):
#     """
#     A class to report various metrics 
#     during training. Maintains an internal 
#     dict for mapping the epoch to the data 
#     gathered. Can be used for either logging 
#     to console or plotting. 
#     """
#     def __init__(self):
#         self._count = 0 
#         self._summ = {}
    
#     def _make_report(self, data):
#         self._summ[self._count] = data
#         self._count += 1 
    
#     def report(self, entry):
#         if not isinstance(entry, dict):
#             raise TypeError("Reporter only accepts dictionary based values atm")
#         self._make_report(entry)
    
#     @property
#     def summary(self):
#         return self._summ

class Reporter(object):
    def __init__(self):
        self._summ = defaultdict(list)
    

    def report(self, ep, name, val):
        if not ep in self._summ.keys():
            self._summ[ep] = [{name:val}] 
        else:
            self._summ[ep].append({name:val})
    

    @property
    def summary(self):
        return self._summ