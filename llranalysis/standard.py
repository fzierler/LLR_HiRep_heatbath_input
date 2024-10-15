import numpy as np
import pandas as pd
import os.path
import shutil
from llranalysis import utils
import re
import matplotlib.pyplot as plt
import tqdm

def BL(P):#Binder cumulant
    return (1 - ( (P.flatten()**4).mean()/ (3 * (((P.flatten()**2).mean())** 2) )  ) ).mean()

def SUS_POLY(P):#Polyakov loop susceptibility
    return ((((np.abs(P).flatten())**2).mean() ) - (np.abs(P).flatten().mean()**2))

def VEV_POLY(P):#VEV of the absolute value of the Polyakov loop
    return np.abs(P).flatten().mean()

def VEV_PLAQ(P):#VEV of the average plaquette
    return P.flatten().mean()

def SH_PLAQ(P):#Specific heat
    return ((((P.flatten())**2).mean() ) - (P.flatten().mean()**2))

def GetCSV(file, file_out, Nb = 1000, M = 1000, plot =True):
    #Creates dataframes containing the values of the observables and the 
    # histogram of the observables
    poly, plaq, beta, V, Lt = ReadOutput(file + 'output_file')
    M_poly = num_blocks(poly)
    M_plaq = num_blocks(plaq)
    VEV_poly = VEV_POLY(poly)
    VEV_poly_err = bootstrap(poly,M_poly, Nb, func = VEV_POLY)
    SUS_poly =  SUS_POLY(poly * ((V / Lt)  ** 0.5)) 
    SUS_poly_err = bootstrap(poly* ((V / Lt)  ** 0.5),M_poly, Nb, func = SUS_POLY)
    VEV_plaq = VEV_PLAQ(plaq)
    VEV_plaq_err = bootstrap(plaq,M_plaq, Nb, func = VEV_PLAQ)
    SH_plaq = SH_PLAQ(plaq) * 6. * V
    SH_plaq_err = bootstrap(plaq,M_plaq, Nb, func = SH_PLAQ)  * 6. * V

    his, bins = np.histogram(plaq,100, density = True)
    bins = bins[:-1] + ((bins[1] - bins[0]) / 2)    
    bets = np.ones(len(his)) * beta
    
    bl = BL(plaq)
    bl_err = bootstrap(plaq,M_plaq, Nb, func = BL)
    
    FULL_DF = pd.DataFrame(data = {'Beta':np.ones(plaq.size) * beta,'Plaq':plaq,'Poly':poly,'V':np.ones(plaq.size) *V,'Lt':np.ones(plaq.size) * Lt})
    os.makedirs(file_out, exist_ok=True)
    FULL_DF.to_csv(file_out + 'full.csv', index = False)
    
    DF= pd.DataFrame(data = {'Beta':beta,'Plaq':VEV_plaq,'Plaq_err':VEV_plaq_err.mean(),'Plaq_SH':SH_plaq, 'Plaq_SH_err':SH_plaq_err.mean(),'Poly':VEV_poly, 'Poly_err':VEV_poly_err.mean(), 'Poly_sus':SUS_poly,'Poly_sus_err':SUS_poly_err.mean(),'Plaq_binder':bl,'Plaq_binder_err':bl_err.mean(), 'Lt':Lt, 'V': V}, index=[beta])
    HIST_DF= pd.DataFrame(data = {'Beta':bets,'Hist':his,'Bins':bins})
    return DF, HIST_DF

def num_blocks(X):
    #calculates the number of blocks for bootstrapping
    leng = X.shape[0]
    avr, f2, f = 0., 0., 0.
    avr = np.mean(X)
    f2 = np.sum(X**2.)
    f = 2. * np.sum(X)
    C0 = (f2 / X.shape[0]) - (avr*avr)
    tint = 0.5
    valid = False
    for M in range(1 , leng):
        f2, f, avr = 0., 0., 0.
        f2 = np.sum(X[:(leng - M)] * X[M:])
        f = np.sum(X[:(leng - M)] + X[M:])
        avr = np.sum(X[:(leng - M)])
        tint += (f2 + (avr*(avr - f)/(leng-M))) / (C0*(leng-M))
        if(M>(4.0*tint)):
            valid = True
            break
    if valid:
        autocorr=np.ceil(tint)
        err=(2.*(2.+M+1.)/leng)*tint
    else:
        autocorr, err = 0.,0.
        print("Error")
    n_block = 1
    for n in range(int(autocorr* 4), leng):
        if(leng %  n== 0):
            n_block = int(leng /  n)
            break
    return(n_block)

def ReadOutput(file):
    #Reads output of HiRep
    txt_beta = 'beta'
    txt_py = "(Polyakov direction 0 =|FUND_POLYAKOV)"
    txt_v = "Global size is "
    new_V = []
    beta = 0.
    poly = []
    plaq = []
    txt_plq = 'Plaquette'
    with open(file, "r") as read_file:
        for line in read_file:
            if(re.findall(txt_beta, line) != []):
                for word in re.split("=", line):
                    if(utils.check_float(word)):
                        beta = float(word)
            elif(re.findall(txt_plq, line) != []):
                if('Configuration' not in line):
                    for word in re.split("\s|:", line):
                        if(utils.check_float(word)):
                            plaq.append(float(word))
            elif((len(re.findall(txt_py, line)) > 1)): 
                    py_tmp = 0.
                    for word in re.split("\s",re.split("=", line)[1]):
                        if(utils.check_float(word)):
                            py_tmp += float(word) ** 2
                    poly.append(py_tmp ** 0.5)
            elif(re.findall(txt_v, line) != []): 
                    for word in re.split("\s",line):
                        for w in re.split("x",word):
                            if(utils.check_float(w)):
                               new_V.append(float(w))
    V = np.prod(new_V)
    Lt = np.min(new_V)
    poly = np.array(poly)
    plaq = np.array(plaq)
    if(beta == 0):
        print('Beta not found')
    return poly, plaq, beta, V, Lt

def bootstrap( P, M, Nb, con_int =0.68, plot = False, func = np.mean):
    #Bootstrap error
    P = P.flatten()
    VEV = func(P)
    block_len = int(np.floor(P.shape[0] / M))
    P = P[P.shape[0] - (M * block_len):].reshape(M, block_len)
    P_func = np.zeros((Nb))
    for i in range(Nb):
        P_func[i] = func(P[np.random.randint(0,M, size = (M)),:])
    P_func.sort()
    return P_func.std(ddof=1)

def CSV(files, files_out, folder_in,folder_out,disable_prog=False):
    # checks if CSV files exists, creates them if not, returns result
    exists = os.path.isfile(folder_in + 'std.csv') and os.path.isfile(folder_in + 'hist.csv') 
    # check if all files named 'full.csv' exist in input directory
    # if all files exist then copy them to the new directory
    for file in tqdm.tqdm(files,disable=disable_prog):
        exists = exists*os.path.isfile(file + "full.csv")

        if exists:
            DF, HIST_DF = ReadCSVFull(folder_in)
            # copy all csv files in individual standard importance sampling runs
            for (ind,file) in enumerate(files):
                os.makedirs(files_out[ind], exist_ok=True)
                file1 = file + "full.csv"
                file2 = files_out[ind] + "full.csv"
                # only copy if input file and output files correspond to different files
                if file1 != file2:
                    shutil.copy2(file1,file2)

        else:
            DF, HIST_DF = ReadDataFull(files,files_out)
        
    # FZ: Always save a copy in the output_directory
    os.makedirs(folder_out, exist_ok=True)
    DF.to_csv(folder_out + 'std.csv', index = False)
    HIST_DF.to_csv(folder_out + 'hist.csv', index = False)
    return DF, HIST_DF

def ReadDataFull(files,files_out):
    #saves CSV files to specified location
    DF, HIST_DF = ReadOutputFull(files,files_out)
    DF = DF.sort_values(by=['Beta'], ignore_index=True)
    return DF, HIST_DF
    
def ReadCSVFull(folder):
    #reads CSV files from specified location
    DF = pd.read_csv(folder + 'std.csv')
    HIST_DF = pd.read_csv(folder + 'hist.csv')
    return DF, HIST_DF

def ReadOutputFull(files,files_out):
    #creates CSV files with all results in
    DF = []
    HIST_DF = []
    for ind, file in enumerate(files):
        file_out = files_out[ind]
        DF_tmp, HIST_DF_tmp = GetCSV(file,file_out)
        DF.append(DF_tmp)
        HIST_DF.append(HIST_DF_tmp)

    DF      = pd.concat(DF, ignore_index=True)
    HIST_DF = pd.concat(HIST_DF, ignore_index=True)
    return DF, HIST_DF

def plot_history(file,N_poly = - 1, N_plaq = -1):
    df_name = file + 'full.csv'
    print(file)
    if os.path.isfile(df_name):
        full_df = pd.read_csv(df_name)
        plt.subplot(2,1,1)
        plt.plot(full_df['Poly'][:N_poly])
        plt.ylabel('$|l_p|$',fontsize=30)
        plt.xlabel('N configurations',fontsize=30)
        plt.subplot(2,1,2)
        plt.hist(full_df['Poly'], 1000)
        plt.xlabel('$|l_p|$',fontsize=30)
        #plt.tight_layout()
        plt.show()
        plt.subplot(2,1,1)
        plt.plot(full_df['Plaq'][:N_plaq])
        plt.ylabel('$u_p$',fontsize=30)
        plt.xlabel('N configurations',fontsize=30)
        plt.subplot(2,1,2)
        plt.hist(full_df['Plaq'], 1000)
        plt.xlabel('$u_p$',fontsize=30)
        #plt.tight_layout()
        plt.show()
    else:
        print(df_name + ' not found')