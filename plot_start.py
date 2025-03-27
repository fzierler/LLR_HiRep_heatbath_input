#!/usr/bin/env python
# coding: utf-8
import llrinput.inputfiles as llr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_input(init_file):
    init_df = pd.read_csv(init_file)
    beta = init_df['beta'] 
    plaq = init_df['plaq']
    umin = np.min(plaq)
    umax = np.max(plaq)
    # Add some data that needs to be interpolate
    DeltaU = (umax-umin)/5
    uks  = np.linspace(umin-DeltaU,umax+DeltaU, 100)
    return beta, plaq, uks 

init_file_std = "input/data/Nt4_std_beta_vs_S0.txt"
init_file_llr = "input/data/4x40_25repeats_128replicas.txt"

# Plot initial guess based on importance sampling data with a fit
beta, plaq, uks = read_input(init_file_std)
aks_fit = llr.fit_initial_an(plaq,beta,uks)
plt.scatter(plaq,beta,label="importance sampling data")
plt.plot(uks,aks_fit,label="fit importance sampling")

# Plot initial guess based on existing llr results with spline interpolation
beta, plaq, uks = read_input(init_file_llr)
aks_int = llr.interpolate_initial_an(plaq,beta,uks)
plt.scatter(plaq,beta,label="llr data")
plt.plot(uks,aks_int,label="interpolate llr")

plt.title("Initial guess for $a_n$")
plt.legend()
plt.show()