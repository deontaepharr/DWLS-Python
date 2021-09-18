import pandas as pd
import numpy as np
import statsmodels.api as sm
from quadprog import solve_qp #https://pypi.org/project/quadprog/ || installed with conda -- look into how/what || #dual method of Goldfarb and Idnani (1982, 1983) for solving quadratic programming problems of the form 

def perform_ols(sig_mat, bulk_sample):
    # NOTES:
    # S = signature matrix ==> sig_mat
    # B = Bulk data individual sample ==> bulk_sample
    # sol = OLS initial weights ==> ols_weights
    sigsig_dot = sig_mat.T.dot(sig_mat).values #D=G
    sigbulk_dot = sig_mat.T.dot(bulk_sample).values #d=a
    sig_eye_mat = np.identity(sig_mat.shape[1]) #A=C
    sig_zeros_mat = np.array([0.0] * sig_mat.shape[1]) #b=x ; TODO: lookup the purpose? ; also must be float: https://stackoverflow.com/questions/36510859/cvxopt-qp-solver-typeerror-a-must-be-a-d-matrix-with-1000-columns
    qprog_sol = solve_qp(G=sigsig_dot, a=sigbulk_dot, C=sig_eye_mat, b=sig_zeros_mat, meq=0, factorized=False)
    qprog_sol = qprog_sol[0] #gets the solution
    qprog_sol = pd.DataFrame(qprog_sol, index=sig_mat.columns) #convert to dataframe, with cell type as index
    
    return qprog_sol

def find_dampening_constant(sig_mat, bulk_sample, ols_weights, epochs=100):
    ws = (1/sig_mat.dot(ols_weights)).pow(2)
    ws_scaled = ws/ws.min()
    ws_val = int(np.ceil(np.log2(ws_scaled.loc[ws_scaled[0] != np.inf, 0].max())))

    solutions_std_list = []
    for j in range(ws_val):
        multiplier = np.power(2, j)
        ws_dampened = ws_scaled.copy()
        ws_dampened.loc[ws_dampened[0] > multiplier, 0] = multiplier
        
        solutions = []
        for i in range(epochs):
            np.random.seed(i)
            n=int(ws_dampened.shape[0]/2)
            subset_indices = np.random.choice(a=ws_dampened.shape[0], size=n, replace=False)
            y = bulk_sample.iloc[subset_indices]
            X = sig_mat.iloc[subset_indices] #Having -1 adds a constant, purpose?IDK: https://stackoverflow.com/questions/52596724/linear-regression-in-r-and-python-different-results-at-same-problem
            w = ws_dampened.iloc[subset_indices]
            wls_model = sm.WLS(y, X, weights=w, missing="raise")
            wls_result = wls_model.fit(method="qr").params
            wls_result = wls_result * (ols_weights.sum() / wls_result.sum()).values[0]
            solutions.append(wls_result)


        solutions_std = pd.concat(solutions, axis=1).std(axis=1)
        solutions_std = solutions_std.rename(j+1) #rename to dampening constant (multiplier)+1
        solutions_std_list.append(solutions_std)

    solutions_std_df = pd.concat(solutions_std_list, axis=1)
    dampening_constant = solutions_std_df.pow(2).mean(axis=0).idxmin() #index=dampening constant (multiplier)
    return dampening_constant

def solve_dampenedWLS_with_constant(sig_mat, bulk_sample, ols_weights, dampening_constant):
    # NOTES:
    # S = signature matrix ==> sig_mat
    # B = Bulk data individual sample ==> bulk_sample
    # sol = goldStandard = OLS initial weights ==> ols_weights
    multiplier = 1 * np.power(2, (dampening_constant-1))
    sol = ols_weights.copy()
    ws = (1 / sig_mat.dot(sol)).pow(2)
    ws_scaled = ws / ws.min()
    ws_dampened = ws_scaled.copy()
    ws_dampened.loc[ws_dampened[0] > multiplier, 0] = multiplier
    W_diag = pd.DataFrame(np.diag(ws_dampened[0]),index=ws_dampened.index,columns=ws_dampened.index)
    D = sig_mat.T.dot(W_diag).dot(sig_mat)
    d = sig_mat.T.dot(W_diag).dot(bulk_sample)
    sig_eye_mat = np.identity(sig_mat.shape[1]) #A
    sig_zeros_mat = np.array([0.0] * sig_mat.shape[1]) #b=x ; TODO: lookup the purpose? ; also must be float: https://stackoverflow.com/questions/36510859/cvxopt-qp-solver-typeerror-a-must-be-a-d-matrix-with-1000-columns
    sc = np.linalg.norm(D, ord=2) #2-norm (largest sing. value) || specifies the “spectral” or 2-norm, which is the largest singular value (svd) of x.
    qprog_sol = solve_qp(G=(D/sc).values, a=(d/sc).values, C=sig_eye_mat, b=sig_zeros_mat, meq=0, factorized=False)
    qprog_sol = qprog_sol[0] #gets the solution
    qprog_sol = pd.DataFrame(qprog_sol, index=sig_mat.columns) #convert to dataframe, with cell type as index
    return qprog_sol

def solve_dampenedWLS(sig_mat, bulk_sample):
    ols_weights = perform_ols(sig_mat, bulk_sample)
    dampening_constant = find_dampening_constant(sig_mat, bulk_sample, ols_weights, epochs=100)
    iterations =0
    changes = []
    change = 1
    solution = solve_dampenedWLS_with_constant(sig_mat, bulk_sample, ols_weights, dampening_constant)
    while change>.01 and iterations<1000 : 
        new_solution = solve_dampenedWLS_with_constant(sig_mat, bulk_sample, ols_weights, dampening_constant)
        solution_average = pd.concat([solution] + [new_solution]*4 , axis=1).mean(1)
        change = np.linalg.norm((solution_average-solution[0]).to_frame(), ord=1)
        solution = solution_average
        iterations += 1
        changes.append(change)
        
    return solution/solution.sum()