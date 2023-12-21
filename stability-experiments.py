import numpy as np
from scipy import stats, optimize

def sig(x):
    return 1 / (1 + np.exp(-x))

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

from collections import defaultdict

import matplotlib.patches as patches

## specify plot options 
plt.rcParams.update({
    'axes.linewidth' : 2,
    'font.size': 22,
    "text.usetex": True,
    'font.family': 'serif', 
    'font.serif': ['Computer Modern'],
    'text.latex.preamble' : r'\usepackage{amsmath,amsfonts}'})

## custom color palette
lblue = (40/255,103/255,178/255)
cred  = (177/255, 4/255, 14/255)

## subsample m out of [n] without replacement B times
def subsample(n, m, B):
    idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        row = np.random.choice(n, m, replace=False)
        idx[b, :] = row
    return(idx)

## fit base algorithm A to B bags; for each
## bag, compute prediction at test point x_te
def fit_bags(Z_tr, x_te, A, m, B):
    n    = len(Z_tr)
    idx  = subsample(n, m, B)
    yhat = np.zeros(B)
    for b in range(B):
        yhat[b] = A(Z_tr[idx[b], :], x_te)
    return(idx, yhat)

## assess stability of subbagging algorithm A 
## on training data Z_tr and test point x_te
## returns sorted list of LOO errors:
##    |\hat{f}(x_te) - \hat{f}^{-i}(x_te)|
def stability(Z_tr, x_te, A, m, B):
    n    = len(Z_tr)
    idx, yhat = fit_bags(Z_tr, x_te, A, m, B)
    y_full = np.mean(yhat)
    loo = []
    for i in range(n):
        # average over bags not containing i
        y_noti = np.mean(yhat[~np.any(idx==i, axis=1)])
        loo.append(abs(y_full - y_noti)) 
    return(np.sort(loo)) # remove y_full

## base algorithm for logistic regression with ridge penalty
## solving 
##    betahat = argmin{ 1/n sum_i loss_i + (lambda/2)*||beta||^2}
## where lambda = 1e-3
def LR(Z_tr, x_te, reg=1e3):
    n = Z_tr.shape[0]
    X_tr, y_tr = Z_tr[:, :-1], Z_tr[:, -1]
    m = LogisticRegression(penalty='l2', 
                           C=reg/n, fit_intercept=False)
    m.fit(X_tr, y_tr)
    return(m.predict_proba(x_te)[0][0])

## base algorithm for regression trees
## where max_depth=50
def fit_tree(Z_tr, x_te):
    n = Z_tr.shape[0]
    X_tr, y_tr = Z_tr[:, :-1], Z_tr[:, -1]

    m = DecisionTreeRegressor(max_depth=50)
    m.fit(X_tr, y_tr)
    return(m.predict(x_te)[0])

def MLP(Z_tr, x_te):
    n = Z_tr.shape[0]
    X_tr, y_tr = Z_tr[:, :-1], Z_tr[:, -1]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(40,),
        max_iter=8,
        alpha=1e-4,
        solver="sgd",
        verbose=0,
        random_state=1,
        learning_rate_init=0.2,
    )
    mlp.fit(X_tr, y_tr)
    return(mlp.predict_proba(x_te)[0][0])


##########################################
#### Stability of logistic regression ####
##########################################
p = 0.5
B = 10000
ns = np.array([1000, 500]) 
d = 200
beta = .1*np.ones(d)

results  = defaultdict(list)

for n in ns:
    # sample from nonnull logistic model
    np.random.seed(1234567891)

    # test
    (np.dot(np.random.randn(1,d), beta))

    x_te = np.random.randn(1,d)
    x_te = x_te-np.mean(x_te)

    X_tr = np.random.randn(n,d)
    y_tr = (np.random.rand(n) < sig(X_tr@beta))*1
    y_tr = y_tr[:, np.newaxis]

    Z_tr = np.hstack([X_tr, y_tr])

    nm = 'LR, n=%d' % n
    print(nm)
    # run base algorithm on full dataset
    y_full = LR(Z_tr, x_te)

    # run base algorithm on LOO datasets
    loo = []
    for i in range(n):
        idx = np.delete(np.arange(n), i)
        y_noti = LR(Z_tr[idx], x_te)
        loo.append(abs(y_full - y_noti)) 
    results[nm] = np.sort(loo)[::-1]

    # run subbagging
    nm = 'Subbagged LR, n=%d' % n
    print(nm)
    results[nm] = stability(Z_tr, x_te, LR, int(n*p), B)[::-1]

#####################################
########### Make Figure 1 ###########
#####################################
fig, ax = plt.subplots(figsize=(6,4), frameon=False)

bins = np.linspace(0, .3, 23)

n = 500
ind = 0
ax.hist(results['LR, n=%d' % n], 
        bins=bins, color=cred, 
        alpha=.6, label='Logistic Regression')  
ax.hist(results['Subbagged LR, n=%d' % n], 
        bins=bins, color=lblue, 
        alpha=.6, label='Logistic Regr. with Subbagging')
plt.xlim([0, .3])

# swap legend handles
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
plt.legend([handles[idx] for idx in order],
           [labels[idx] for idx in order], 
           prop={'size':16})

plt.xlabel('Leave-one-out error $|\\hat{f}(x) - \\hat{f}^{\\setminus i}(x)|$')

plt.tight_layout()
plt.savefig('fig1.pdf')

##########################################
#### NN on logistic regression data   ####
##########################################
warnings.filterwarnings("ignore", 
                        category=ConvergenceWarning, 
                        module="sklearn")

nm = 'Subbagged NN, n=%d' % n
print(nm)
results[nm] = stability(Z_tr, x_te, MLP, int(n*p), B)[::-1]

nm = 'NN, n=%d' % n
print(nm)
y_full = MLP(Z_tr, x_te)
loo = []
for i in range(n):
    idx = np.delete(np.arange(n), i)
    loo.append(abs(y_full - MLP(Z_tr[idx], x_te))) 
results[nm] = np.sort(loo)[::-1]

#####################################
#### Stability of decision trees ####
#####################################
np.random.seed(1234567891)
n, d = 500, 40
X = np.random.rand(n, d)
y = np.zeros(n)
for j in range(d):
    y += np.sin(X[:, j]/(1+j))
y[::4] += 2*(.5-np.random.rand(int(n/4)))
y[::3] += .5*(.5-np.random.rand(1+n//3))

x_te = np.random.rand(1, d)

Z = np.hstack([X, y[:, np.newaxis]])

nm = 'DT, n=%d' % n
print(nm)
yhat = fit_tree(Z, x_te)

loo, rs = [], []
for i in range(n):
    idx = np.delete(np.arange(n), i)
    y_noti = fit_tree(Z[idx], x_te)
    loo.append(abs(yhat - y_noti))
results[nm] = np.sort(loo)[::-1]

nm = 'Subbagged DT, n=%d' % n
print(nm)
results[nm] = stability(Z, x_te, fit_tree, int(n*p), B)[::-1]

#####################################
########### Make Figure 3 ###########
#####################################
import matplotlib.transforms as mtrans

NBINS = 80

### make small plots
d = 200
fig, axs = plt.subplots(4, 2, figsize=(14,16), frameon=False)

for ind, n in enumerate([500, 1000]):
    bins = np.linspace(0, .5, NBINS)

    axs[ind,0].hist(results['LR, n=%d' % n], 
                    bins=bins, color=cred, alpha=.6, 
                    label='Base algorithm $\\mathcal{A}$')  
    axs[ind,0].hist(results['Subbagged LR, n=%d' % n], 
                    bins=bins, color=lblue, alpha=.6, 
                    label='Subbagged algorithm $\\widetilde{\\mathcal{A}}_{B}$')

    axs[ind,1].plot(results['LR, n=%d' % n], 
                    np.arange(n)/n, lw=3, c=cred, 
                    label='Base algorithm $\\mathcal{A}$')
    axs[ind,1].plot(results['Subbagged LR, n=%d' % n], 
                    np.arange(n)/n, lw=3, c=lblue, 
                    label='Subbagged algorithm $\\widetilde{\\mathcal{A}}_{B}$')
    k = 10000
    delta = np.arange(k)/k
    axs[ind,1].plot(1/(4*delta*n)**.5, delta, 'k', ls='dotted', 
                    label='Stability guarantee\nfor subbagging', lw=3)

    axs[ind,0].set_xlim([0, .25])

    axs[ind,1].set_ylabel('Error probability $\\delta$')
    axs[ind,0].set_ylabel('Frequency')

    axs[ind,1].set_ylim([0, .5])
    axs[ind,1].set_xlim([0.0, 10/np.sqrt(500)])
    axs[ind,1].legend(fontsize=16)
    axs[ind,0].legend(fontsize=16)

n,d = 500,200
bins = np.linspace(0, .5, NBINS)
axs[2,0].hist(results['NN, n=%d' % n], 
              bins=bins, color=cred, alpha=.6, 
              label='Base algorithm $\\mathcal{A}$')  
axs[2,0].hist(results['Subbagged NN, n=%d' % n], 
              bins=bins, color=lblue, alpha=.6, 
              label='Subbagged algorithm $\\widetilde{\\mathcal{A}}_{B}$')

axs[2,0].set_xlim([0, .25])
axs[2,1].plot(results['NN, n=%d' % n], 
              np.arange(n)/n, lw=3, c=cred, 
              label='Base algorithm')
k = 10000
delta = np.arange(k)/k
axs[2,1].plot(results['Subbagged NN, n=%d' % n], 
              np.arange(n)/n, lw=3, c=lblue, 
              label='Subbagging')
axs[2,1].plot(1/(4*delta*n)**.5, delta, 'k', ls='dotted', lw=3, 
              label='Stability guarantee\nfor subbagging')

axs[2,1].set_ylabel('Error probability $\\delta$')
axs[2,0].set_ylabel('Frequency')

axs[2,1].set_ylim([0, .5])
axs[2,1].set_xlim([0.0, 10/np.sqrt(n)])
axs[2,1].set_yticks([])
    
n,d = 500,40
bins = np.linspace(0, .5, NBINS)
axs[3,0].hist(results['DT, n=%d' % n], 
              bins=bins, color=cred, alpha=.6, 
              label='Base algorithm $\\mathcal{A}$')  
axs[3,0].hist(results['Subbagged DT, n=%d' % n], 
              bins=bins, color=lblue, alpha=.6, 
              label='Subbagged algorithm $\\widetilde{\\mathcal{A}}_{B}$')

axs[3,0].set_xlim([0, .25])
axs[3,1].plot(results['DT, n=%d' % n], 
              np.arange(n)/n, lw=3, c=cred, 
              label='Base algorithm')
k = 10000
delta = np.arange(k)/k
axs[3,1].plot(results['Subbagged DT, n=%d' % n], 
              np.arange(n)/n, lw=3, c=lblue, 
              label='Subbagging')
axs[3,1].plot(1/(4*delta*n)**.5, delta, 'k', ls='dotted', lw=3, 
              label='Stability guarantee\nfor subbagging')

axs[3,1].set_ylabel('Error probability $\\delta$')
axs[3,0].set_ylabel('Frequency')

axs[3,1].set_ylim([0, .5])
axs[3,1].set_xlim([0.0, 10/np.sqrt(n)])
    
fig.tight_layout()
plt.subplots_adjust(hspace = .7)

# Get the bounding boxes of the axes including text decorations
r = fig.canvas.get_renderer()
get_bbox = lambda ax: (ax
                       .get_tightbbox(r)
                       .transformed(fig.transFigure.inverted()))
bboxes = np.array(list(map(get_bbox, axs.flat)), mtrans.Bbox).reshape(axs.shape)

#Get the minimum and maximum extent, get the coordinate half-way between those
ymax = (np.array(list(map(lambda b: b.y1, bboxes.flat)))
        .reshape(axs.shape)
        .max(axis=1))
ymin = (np.array(list(map(lambda b: b.y0, bboxes.flat)))
        .reshape(axs.shape)
        .min(axis=1))
ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)
ys = ymin[:-1]

######## 
for i in range(4):
    axs[i,1].set_xlabel('Error tolerance $\\varepsilon$')
    axs[i,1].legend(fontsize=16)
    axs[i,0].legend(fontsize=16)
for i in range(3):
    axs[i,0].set_xlabel('Leave-one-out perturbation ' + 
                        '$|\\hat{f}(x) - \\hat{f}^{\\setminus i}(x)|$')
axs[3,0].set_xlabel('Leave-one-out perturbation ' + 
                    '$\\frac{|\\hat{f}(x) - \\hat{f}^{\\setminus i}(x)|}' + 
                    '{\\text{Range}(\\mathcal{D},x)}$')

titles = ['Setting 1: Logistic regression ($n=500,d=200$)', 
          'Setting 2: Logistic regression ($n=1000,d=200$)', 
          'Setting 3: Neural network ($n=500,d=200$)', 
          'Setting 4: Regression trees ($n=500,d=40$)']
for i, title in enumerate(titles):
    plt.figtext(0.5, ymax[i]+.03, title, ha="center", va="top", fontsize=26)

axs[2,1].set_xlabel('Error tolerance $\\varepsilon$')
axs[2,1].set_yticks([0.0, 0.2, 0.4])

plt.savefig('fig3.pdf', bbox_inches='tight')

#####################################
########### Make Figure 2 ###########
#####################################
from matplotlib.colors import LogNorm

fig = plt.figure(figsize=(9,9), frameon=False)

p = .5
dd = np.linspace(0, 1, 1000)
# upper bound from theorem 3.4
UPPER = np.sqrt((1/(n-1))*(1/dd)*p/(1-p)*(1/(4)))
plt.plot(UPPER, dd, '-', c=lblue, alpha=1, lw=3)
plt.fill_between(UPPER, dd, np.repeat(dd.max(), len(dd)), color=lblue, alpha=.1)

n = 500
m = int(n*p)
# lower bound from theorem 3.7
LOWER = (p*(1-dd-1/n)*
         stats.hypergeom.pmf(
             np.floor(p*(1+np.floor(n*dd))), 
             n-1, m, np.floor(n*dd)))
plt.plot(LOWER, dd, '-', c=cred, alpha=1, lw=3)
plt.fill_between(LOWER, np.repeat(0, len(dd)), dd, color=cred, alpha=.1)

plt.text(2.5/np.sqrt(n), .15, 
         'Derandomized bagging\nis ' + 
         '$(\\varepsilon, \\delta)$-stable for\nany base algorithm.', c=lblue)
plt.text(.12/np.sqrt(n), .009, 
         'Derandomized bagging\nis not ' + 
         '$(\\varepsilon, \\delta)$-stable\nfor some base algorithm.', c=cred)

plt.xlim([0, 5/np.sqrt(n)])
plt.ylim([0, .2])
plt.yticks([0,.05,.1,.15,.2])
plt.ylabel('Error probability $\\delta$', fontsize=26)

plt.xlabel('Error tolerance $\\varepsilon$', fontsize=26)
plt.tight_layout()
plt.savefig('fig2.pdf')