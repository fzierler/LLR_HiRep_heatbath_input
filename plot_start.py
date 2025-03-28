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
    DeltaU = (umax-umin)/20
    uks  = np.linspace(umin-DeltaU,umax+DeltaU, 100)
    return beta, plaq, uks 

init_file_std_Nt4 = "input/data/Nt4_std_beta_vs_S0.txt"
init_file_llr_Nt4 = ["input/data/4x40_25repeats_128replicas.txt"]

init_file_std_Nt5 = "input/data/Nt5_std_beta_vs_S0.txt"
init_file_llr_Nt5 = ["input/data/5x72_10repeats_95replicas.txt"]

init_file_std_Nt6 = "input/data/Nt6_std_beta_vs_S0.txt"
init_file_llr_Nt6 = ["input/data/6x72_25repeats_48replicas.txt"]

init_file_std = init_file_std_Nt6
init_file_llr = init_file_llr_Nt6

# Plot initial guess based on existing llr results with spline interpolation
for f in init_file_llr:
    beta, plaq, uks = read_input(f)
    aks_int = llr.interpolate_initial_an(plaq,beta,uks)
    plt.scatter(plaq,beta,label="llr data")
    plt.plot(uks,aks_int,label="interpolate llr")

# Plot initial guess based on importance sampling data with a fit
beta, plaq, uks = read_input(init_file_std)
aks_int = llr.interpolate_initial_an(plaq,beta,uks)
plt.plot(uks,aks_int,label="interpolate importance sampling")
aks_fit = llr.fit_initial_an(plaq,beta,uks)
plt.plot(uks,aks_fit,label="fit importance sampling")
plt.scatter(plaq,beta,label="importance sampling data")

plt.title("Initial guess for $a_n$")
plt.legend()
plt.show()