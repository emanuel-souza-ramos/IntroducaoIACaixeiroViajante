from objective_function_tsp import objective_function_tsp
def two_opt_best_improvement(n,s,f,D):
    s_current = s.copy()
    s_best = s.copy()
    f_best = f
    for i in range(n-1):
        for j in range(1,n):
            s_candidate = s_current.copy()
            s_candidate[i],s_candidate[j]=s_candidate[j],s_candidate[i]
            f_new = objective_function_tsp(n,s_candidate,D)
            if f_new < f_best:
                s_best = s_candidate.copy()
                f_best = f_new
                s_current = s_candidate.copy()
    return(s_best,f_best)

