## this file is test for python language

import tensorflow as tf
import numpy as np

def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max:   
        a, b = b, a + b 
        n = n + 1 
        yield n

