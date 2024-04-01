# Utility functions for plotting

def update_max(numlist=None, current_max=None):
    '''
    Method to return the maximum value of the numlist. If numlist is None and current_max 
    is None, then default is 0.
    '''

    if numlist is None and current_max is None:
        return 0

    elif numlist is None and current_max is not None:
        return current_max
    
    else:
        return max(current_max, max(numlist))
