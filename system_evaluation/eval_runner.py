'''
Script that runs the system-level evaluation for different types of confusion matrices

Notes:
1. Hard to store the same controller since initial conditions keep changing. However, it might be better to store one controller, 
and have the Markov chain regenerate. This is critical since each Markov chain method is repeating 
the controller computations, which might lead to unfair comparisons between different types of confusion matrices.

2. Infact, even for the same specification, there could be different types of controllers resulting in
inconsistent evaluations. 
'''

