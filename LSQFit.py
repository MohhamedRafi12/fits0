import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

xmin=1.0
xmax=20.0
npoints=12
sigma=0.2
lx=np.zeros(npoints)
ly=np.zeros(npoints)
ley=np.zeros(npoints)
pars=[0.5,1.3,0.5]

from math import log
def f(x,par):
    return par[0]+par[1]*np.log(x)+par[2]*np.log(x)*np.log(x)

from random import gauss
def getX(x):  # x = array-like
    step=(xmax-xmin)/npoints
    for i in range(npoints):
        x[i]=xmin+i*step
        
def getY(x,y,ey):  # x,y,ey = array-like
    for i in range(npoints):
        y[i]=f(x[i],pars)+gauss(0,sigma)
        ey[i]=sigma

# # get a random sampling of the (x,y) data points, rerun to generate different data sets for the plot below
# getX(lx)
# getY(lx,ly,ley)

# fig, ax = plt.subplots()
# ax.errorbar(lx, ly, yerr=ley)
# ax.set_title("Pseudoexperiment")
# fig.show()


# *** modify and add your code here ***
nexperiments = int(1e6)  # for example

# we want to implement linear fit matrix
# define X = [1, ln x_1, ln(x_1)^2], p = [a,b,c]^T 
# y = Xp (model), 
# weight matrix is W_ii = 1/sigma_i^2 (diagonal)
# we minimize (X^T W X)p = X^T W y  

def xmatrix(x): 
    lx = np.log(x)
    return np.column_stack([np.ones_like(lx), lx, lx**2]) 

def wls_fit(x, y, ey): 
    X = xmatrix(x) # (n, 3)
    w = 1 / (ey * ey) # (n, ) 

    # normal equations, 
    XT_W = (X.T * w) # (3, n)
    A = XT_W @ X # (3, 3) # (matrix multiplication ==> @)
    b = XT_W @ y # (3, )
    p = np.linalg.solve(A, b) # [a, b, c]
    
    yhat = X @ p 
    chi2 = np.sum(((y-yhat)/ey)**2)
    dof = len(y) - X.shape[1]
    return p, chi2, dof


rng = np.random.default_rng(12345) 

par_a = np.random.rand(nexperiments)   
par_b = np.random.rand(nexperiments)   
par_c = np.random.rand(nexperiments)
chi2_reduced = np.random.rand(nexperiments)

x = np.empty(npoints, dtype=float)
getX(x)

for i in range(nexperiments):
    y = np.empty(npoints, dtype=float)
    ey = np.empty(npoints, dtype=float) 


    # generate data
    y[:] = f(x, pars) + rng.normal(0.0, sigma, size=npoints) 
    ey[:] = sigma

    p, chi2, dof = wls_fit(x, y, ey)
    par_a[i], par_b[i], par_c[i] = p
    chi2_reduced[i] = chi2 / dof


x = np.empty(npoints)
y = np.empty(npoints)
ey = np.empty(npoints)

getX(x)
getY(x, y, ey)

p, chi2, dof = wls_fit(x, y, ey)
print(f"Fitted parameters: a={p[0]:.4f}, b={p[1]:.4f}, c={p[2]:.4f}")
print(f"$\chi^2$/ndf = {chi2/dof:.3f}")

# make smooth curve for plotting
xs = np.linspace(xmin, xmax, 400)
yfit = f(xs, p)
ytrue = f(xs, pars)

fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(x, y, yerr=ey, fmt='o', label='Pseudo-data')
ax.plot(xs, yfit, label='WLS fit')
ax.plot(xs, ytrue, '--', label='Truth')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f"Least-Squares Fit ($\chi^2$/ndf = {chi2/dof:.2f})")
ax.legend()
fig.savefig('./LSQFit_py_example.pdf')

fig, axs = plt.subplots(2, 2, figsize=(10,10))
plt.tight_layout()

# careful, the automated binning may not be optimal for displaying your results!
h1 = axs[0, 0].hist2d(par_a, par_b, bins=60,)
axs[0, 0].set_xlabel('a'); axs[0, 0].set_ylabel('b'); axs[0, 0].set_title('Parameter b vs a')
fig.colorbar(h1[3], ax=axs[0,0])

h2 = axs[0, 1].hist2d(par_a, par_c, bins=60,)
axs[0, 1].set_xlabel('a'); axs[0, 1].set_ylabel('c'); axs[0, 1].set_title('Parameter c vs a')
fig.colorbar(h2[3], ax=axs[0,1])

h3 = axs[1, 0].hist2d(par_b, par_c, bins=60,)
axs[1, 0].set_xlabel('b'); axs[1, 0].set_ylabel('c'); axs[1, 0].set_title('Parameter c vs b')
fig.colorbar(h3[3], ax=axs[1,0])

axs[1, 1].hist(chi2_reduced, bins=80, density=True)
axs[1, 1].set_xlabel('$\chi^2$/ndf'); axs[1, 1].set_title('Reduced $\chi^2$ distribution')
plt.tight_layout()
# fig.savefig('./LSQFit.pdf')

from matplotlib.backends.backend_pdf import PdfPages

# Create multi-page PDF
with PdfPages("LSQFit_py.pdf") as pdf:
    # First figure: pseudoexperiment
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.errorbar(lx, ly, yerr=ley, fmt='o', label='Pseudo-data')
    xs = np.linspace(xmin, xmax, 400)
    p, chi2, dof = wls_fit(lx, ly, ley)
    ax1.plot(xs, f(xs, p), label='WLS fit')
    ax1.plot(xs, f(xs, pars), '--', label='Truth')
    ax1.set_title(f"Pseudoexperiment (χ²/ndf = {chi2/dof:.2f})")
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.legend()
    pdf.savefig(fig1)        # save this figure as page 1
    plt.close(fig1)

    # Second figure: parameter study (the 2×2 grid)
    fig2, axs = plt.subplots(2, 2, figsize=(9, 8))
    plt.tight_layout(pad=2.0)
    axs[0,0].hist2d(par_a, par_b, bins=60, range=[(0.2,0.8),(0.7,1.9)])
    axs[0,0].set_title('Parameter b vs a')
    axs[0,1].hist2d(par_a, par_c, bins=60, range=[(0.2,0.8),(0.2,0.9)])
    axs[0,1].set_title('Parameter c vs a')
    axs[1,0].hist2d(par_b, par_c, bins=60, range=[(0.7,1.9),(0.2,0.9)])
    axs[1,0].set_title('Parameter c vs b')
    axs[1,1].hist(chi2_reduced, bins=80, density=True)
    axs[1,1].set_title('Reduced χ² distribution')
    for ax in axs.flat:
        ax.grid(True, alpha=0.25)
    fig2.suptitle('Least-squares study results')
    pdf.savefig(fig2)        # save as page 2
    plt.close(fig2)

# **************************************
