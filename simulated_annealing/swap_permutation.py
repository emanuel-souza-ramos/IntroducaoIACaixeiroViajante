import random
def swap_permutation(s,n):
    s_new = s.copy()
    k = random.sample(range(1,n),2)
    s_new[k[0]],s_new[k[1]]=s_new[k[1]],s_new[k[0]]
    return(s_new)