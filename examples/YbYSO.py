import numpy as np
import dieke
import matplotlib.pyplot as plt

nf = 13
Yb = dieke.RareEarthIon(nf)

# simulation params
cfparams = dieke.readLaF3params(nf)
numLSJ = Yb.numlevels()
numLSJmJ = Yb.numstates()

# free ion params (Yb only has zeta)
cfparams['ZETA'] = 2892.4  # this is just an approximate

# crystal field params (of Yb:YSO from Alizadeh PhD thesis, site 1)
cfparams['B20'] = -623
cfparams['B21'] = -601.6-341.0j
cfparams['B22'] = -90.6-316.8j
cfparams['B40'] = 973.7
cfparams['B41'] = 108.6-822.1j
cfparams['B42'] = -765.9-615.9j
cfparams['B43'] = 111.6+31.9j
cfparams['B44'] = -177.0-92.8j
cfparams['B60'] = 374
cfparams['B61'] = -494.6-65.1j
cfparams['B62'] = 350.7-400.3j
cfparams['B63'] = 508.6-54.5j
cfparams['B64'] = 439.3+690.9j
cfparams['B65'] = 196.7+271.9j
cfparams['B66'] = 325.7+283.9j

# crystal field params (of Yb:YSO from Zhou et al, site 1)
# cfparams['B20'] = -507.1
# cfparams['B21'] = 689.4+244.8j
# cfparams['B22'] = 164.5+2.2j
# cfparams['B40'] = 1641.5
# cfparams['B41'] = 308+334.8j
# cfparams['B42'] = -164.2+298.3j
# cfparams['B43'] = -87-15.7j
# cfparams['B44'] = -989.3-246.3j
# cfparams['B60'] = -86
# cfparams['B61'] = 486.2+190.2j
# cfparams['B62'] = 43.3+25.3j
# cfparams['B63'] = 94.1+188.9j
# cfparams['B64'] = 373.9-84.3j
# cfparams['B65'] = 6.5+51.7j
# cfparams['B66'] = -104.4-354.8j

# crystal field params (of Yb:YSO from Alizadeh, site 2)
# cfparams['B20'] = 235.4
# cfparams['B21'] = -388.2+227.6j
# cfparams['B22'] = -468.5-51j
# cfparams['B40'] = 1070.5
# cfparams['B41'] = 790.9+200.4j
# cfparams['B42'] = -240.9-341.4j
# cfparams['B43'] = -101.3+14.8j
# cfparams['B44'] = -373.5+317.2j
# cfparams['B60'] = 277
# cfparams['B61'] = -54.4+60.3j
# cfparams['B62'] = -80.8-2.3j
# cfparams['B63'] = -55.4+214.9j
# cfparams['B64'] = 114.8-121.8j
# cfparams['B65'] = 164.8-40.9j
# cfparams['B66'] = 95.2-98.5j

# crystal field params (of Yb:YSO from Zhou et al, site 2)
# cfparams['B20'] = 133.7
# cfparams['B21'] = -454.6+25j
# cfparams['B22'] = -458.2+100.6j
# cfparams['B40'] = 313
# cfparams['B41'] = 587.4+488.1j
# cfparams['B42'] = -544+607.3j
# cfparams['B43'] = -508.6-172.6j
# cfparams['B44'] = -14-55.4j
# cfparams['B60'] = 269.7
# cfparams['B61'] = 95.8-190.4j
# cfparams['B62'] = -102.7+10.2j
# cfparams['B63'] = -180.7-154.1j
# cfparams['B64'] = 102.9-95j
# cfparams['B65'] = 157.1-311.7j
# cfparams['B66'] = 419.4-249.6j

# Simulation
# Make a free-ion Hamiltonian
H0 = np.zeros([numLSJmJ, numLSJmJ])
for k in cfparams.keys():
    if k in Yb.FreeIonMatrix:
#         print("Adding free ion parameter \'%s\' = %g" % (k, cfparams[k]))
        H0 = H0 + cfparams[k] * Yb.FreeIonMatrix[k]

# Add in the crystal field terms and diagonalise the result
H = H0
for k in [2, 4, 6]:
    for q in range(0, k + 1):
        if 'B%d%d' % (k, q) in cfparams:
            if q == 0:
                H = H + cfparams['B%d%d' % (k, q)] * Yb.Cmatrix(k, q)
            else:
                Bkq = cfparams['B%d%d' % (k, q)]
                Bkmq = (-1) ** q * np.conj(Bkq)
                Ckq = Yb.Cmatrix(k, q)
                Ckmq = Yb.Cmatrix(k, -q)
                #See page 44, eq 3.1 of the crystal field handbook
                H = H + Bkq * Ckq + Bkmq * Ckmq

                
# Diagonalise the Hamiltonian
evals, vec = np.linalg.eig(H)
idx = np.argsort(evals)
evals = evals[idx]
E0 = evals[0] # ground state energy
evals = evals - E0 # make everything relative to GS
vec = vec[:, idx] # sort eigenvalues

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(2,8))
zeroB = np.real(evals)

# site 1 experimental values from literature
ground = [0, 234, 612, 970]
excited = [10216, 10505, 11076]

# site 2 experimental values from literature
# ground = [0, 111, 499, 709]
# excited = [10189, 10391, 10874]

# Excited state manifold plot
ax1.set_title('Alizadeh, Site 1', pad=5)

line1 = ax1.hlines(zeroB, -1, 1, colors='r')
ax1.hlines(excited, -1, 1, colors='k', ls='--')
ax1.set_ylim(1.01e4, 1.150e4)

ax1.axes.get_xaxis().set_visible(False)
ax1.spines['left'].set_color('red')
ax1.tick_params(colors='red', which='both')

# Ground state manifold plot
line2 = ax2.hlines(zeroB, -1, 1)
line3 = ax2.hlines(ground, -1, 1, colors='k', ls='--')
ax2.set_ylim(-100, 1065)

ax2.axes.get_xaxis().set_visible(False)
ax2.spines['left'].set_color('C0')
ax2.tick_params(colors='C0', which='both')

# General plot settings
plt.subplots_adjust(hspace=.0)
ax1.set_ylabel(r'Energy (cm$^{-1}$)')
ax1.yaxis.label.set_color('red')
ax2.set_ylabel(r'Energy (cm$^{-1}$)')
ax2.yaxis.label.set_color('C0')

ax1.legend([line1, line2, line3], ['Exp (excited)', 'Exp (ground)', 'Theory'], loc='upper left', prop={'size': 10})
plt.show()
