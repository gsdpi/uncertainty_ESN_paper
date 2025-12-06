
##################################################################
# 1 ) Get source data: dataicann
# Download from http://hdl.handle.net/10651/53461
# 
# Note: This requires some extra packages (requests
# and zipfile).
# You can also manually download and uncompress the data.
###################################################################

from pathlib import Path

file_path = Path('./dataicann.zip')

if not file_path.exists():   
    # Start the download    
    print('downloading dataset ...')
    import requests
    url = 'https://digibuo.uniovi.es/dspace/bitstream/handle/10651/53461/dataicann.zip?sequence=1&isAllowed=y'
    r = requests.get(url)
    with open('dataicann.zip', 'wb') as outfile:
        outfile.write(r.content)
    print('download completed')
else:
    print('File already exists, skipping download.')
    
import zipfile
with zipfile.ZipFile('./dataicann.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

##################################################################
# 2 ) Reproduce the paper results
##################################################################


import reservoirpy as rpy
from packaging.version import Version

if( Version(rpy.__version__) < Version("0.4")):
    rpy.verbosity(0)  # no need to be too verbose here

rpy.set_seed(42)  # make everyhting reproducible !

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# For article graphics set font to 24 points
plt.rcParams.update({'font.size': 24})

# Read data
PATH = './dataicann/'
d = loadmat(PATH+'dataicann.mat')

#####################################################################
# STEP 1: TRAIN THE ESN WITH A SUBSET OF ACTIVITIES
# In this case, train with fixed values of the resistance added
# to phase R to simulate an electrical defect
#####################################################################
X = []
Y = []
ohm     = [0,5,10,15,20]
train   = [2,6,3,4,5]

for i in range(len(train)):
    paq = d['z'][0][train[i]][:,1]
    paq = paq[:10000]  # use only first half of the signal for training
    X.append(paq)
    Y.append(np.repeat(ohm[i],len(paq)))
X_train = np.hstack(X).reshape(-1,1)
Y_train = np.hstack(Y).reshape(-1,1)


from reservoirpy.nodes import Reservoir, Ridge, Input

n_states = 300
rho=1.270074061545781 
sparsity=0.01
Lr=0.27031482024950293
Win_scale=0.8696730804425951
Wfb_scale=.0
input_scale = 1
Washout = 0
Warmup = 20 #100
set_bias = True # input_bias for the Ridge minimization, if true bias is added to inputs
ridge = 5.530826061879047e-08

print('Creating ESN...')
data = Input()
reservoir = Reservoir(n_states, lr=Lr, sr=rho, input_scaling=input_scale, rc_connectivity=sparsity, Win=rpy.mat_gen.bernoulli(input_scaling = Win_scale))
if(Version(rpy.__version__) < Version("0.4")):
    readout = Ridge(ridge = ridge, input_bias = set_bias)
else:
    readout = Ridge(ridge = ridge, fit_bias = set_bias)
esn_model =  data >> reservoir >> readout
if( Version(rpy.__version__) < Version("0.4")):
    print(esn_model.node_names)
else:
    print(esn_model.nodes)

# Train the ESN with the train data
print('Training ESN...')
esn_model = esn_model.fit(X_train, Y_train, warmup=Warmup)
if( Version(rpy.__version__) < Version("0.4")):
    print(reservoir.is_initialized, readout.is_initialized, readout.fitted)
else:
    print(reservoir.initialized, readout.initialized)


#####################################################################
# STEP 2: TEST THE METHOD - CREATE THE APPROXIMATED PDF
#####################################################################

C_pdf = []
T_pdf = []

print('Running train signal over reservoir...')
states = reservoir.run(X_train)

Q = X_train.shape[0]
tm = 1/5000.
t = np.arange(Q).reshape(-1,1)*tm
S = 200 
L = 1000 

print(f'Decomposing (wnd siz {L}, stride {S})...')
rango = np.arange(0,Q-L,S)
for i in rango:
    idx = np.arange(i,i+L)
    print(f"\rWindow {i} of {rango[-1]}", end='', flush=True)
    # Perform the SVD
    U, s, VT = np.linalg.svd(states[idx,:].T, full_matrices=False)
    # Add the new high-dimensional point
    T_pdf.append(t[idx[0],0])
    C_pdf.append(s)
C_pdf = np.array(C_pdf)
T_pdf = np.array(T_pdf)
print('\nDone...')


#####################################################################
# STEP 3: TEST THE METHOD
#####################################################################

print('Running all signals on the ESN model...')

# Get all signal in vector X
X = []
test   = train + [7,8,0,1]
for i in range(len(test)):
    paq = d['z'][0][test[i]][:,1]
    X.append(paq)
X = np.hstack(X).reshape(-1,1)

states = reservoir.run(X)
Y_t = readout.run(states)

# Apply the sliding window algorithm
C = []
T = []
Y = []
Q = X.shape[0]
tm = 1/5000.
t = np.arange(Q).reshape(-1,1)*tm
S = 500
L = 1000 

print(f'Decomposing (wnd siz {L}, stride {S})...')
rango = np.arange(0,Q-L,S)
for i in rango:
    idx = np.arange(i,i+L)
    print(f"\rWindow {i} of {rango[-1]}", end='', flush=True)
    U, s, VT = np.linalg.svd(states[idx,:].T, full_matrices=False)
    T.append(t[idx[0],0])
    C.append(s)
    Y.append(np.median(Y_t[idx]))
C = np.array(C)
T = np.array(T)
Y = np.array(Y)
print('\nDone...')


# Now let's create the estimated PDFs and calculate the epistemic uncertainty
# score performance for different values of the dimensionality r

# Get classes (seen/unseen)

Classes_ = []
seen = train+[7,8]
unseen = [0,1]
for i in range(len(seen)):
    siz = len(d['z'][0][seen[i]][:,1])
    Classes_.append(np.ones(siz))
for i in range(len(unseen)):
    siz = len(d['z'][0][unseen[i]][:,1])
    Classes_.append(np.zeros(siz))
Classes_ = np.hstack(Classes_).reshape(-1,1)

# Estimate the PDF with KDE for different values of
# dimensionality r, and evaluate the classification
# performance of the score

from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KernelDensity

for r in np.arange(1,21,1):
    values = np.stack(C_pdf[:,0:r])
    bw = len(values)  ** (-1. / (r + 4)) # Scott's rule of thumb
    kernel = KernelDensity(kernel='gaussian', bandwidth=bw).fit(values)
    logprobX = kernel.score_samples(C[:,0:r])
    logprobX_exp = np.kron(logprobX,np.ones(S))
    
    Classes=Classes_[:len(logprobX_exp)]
    
    fpr, tpr, thresholds = roc_curve(Classes, logprobX_exp)
    roc_auc = auc(fpr, tpr)

    rocs_to_plot = [1,2,3,5,10,20]

    if(r in rocs_to_plot):
        plt.figure(1)
        plt.plot(fpr, tpr, label=f'r={r}, AUC = {roc_auc:.3f}', linewidth=2.5, marker = 'o')
        plt.grid(visible=True)
        plt.legend()

    th_optimal = thresholds[np.argmax(tpr - fpr)]
    print(f'Dimensions: {r}')
    print(f'Optimal threshold: {th_optimal}')
    print(f'AUC: {roc_auc}')

    from sklearn.metrics import recall_score, precision_score, f1_score
    sensitivity = recall_score(Classes , logprobX_exp>th_optimal)
    specificity = recall_score(np.logical_not(Classes) , np.logical_not(logprobX_exp>th_optimal))
    precision  = precision_score(Classes, logprobX_exp>th_optimal)
    f1 = f1_score(Classes , logprobX_exp>th_optimal)
    print(f' Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, Precision: {precision:.3f}, F1-score: {f1:.3f}')

    washout = 600
    X_exp = X[washout:len(logprobX_exp)]
    Y_exp = Y_t[washout:len(logprobX_exp)]
    T_exp = t[washout:len(logprobX_exp)]
    logprobX_exp = logprobX_exp[washout:]

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(T_exp,X_exp)
    plt.grid()
    plt.ylabel('acceleration signal')
  
    # Use the optimal threshold to decide the class
    plt.subplot(2,1,2)
    plt.ylabel('Resistance (ohms)')
    plt.xlabel('time (s)')
    
    cc = np.array([1 if i>=th_optimal else 0 for i in logprobX_exp])

    # Masks to decide colors
    mask_green = cc == 0
    mask_red = cc == 1

    # Auxiliary function to plot the segments
    def plot_segments(T, Y, mask, color):
        Y_aux = Y.copy()
        Y_aux[mask] = np.nan #  Asign NaN to the values we want to hide
        plt.plot(T, Y_aux, color=color, alpha=0.8, linewidth=2)

    # Plot segments with different colors
    plot_segments(T_exp, Y_exp, mask_red, 'red')
    plot_segments(T_exp, Y_exp, mask_green, 'green')

    # gray line
    plt.plot(T_exp, Y_exp, color='gray', alpha=0.3)
    plt.grid()


plt.show()



