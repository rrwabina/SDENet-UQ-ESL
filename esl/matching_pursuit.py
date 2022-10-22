import h5py    
import numpy as np    
import matplotlib.pyplot as plt 

directory = 'D:/SimNIBS-3.2/matlab/leadfield/T1W-012_leadfield_EEG10-10_UI_Jurak_2007.hdf5'
leadfield = h5py.File(directory, 'r')
leadfieldmatrix = leadfield['mesh_leadfield']['leadfields']['tdcs_leadfield']
leadfieldmatrix = leadfieldmatrix[:]
leadfieldmatrix = leadfieldmatrix.reshape((leadfieldmatrix.shape[0], leadfieldmatrix.shape[1] * leadfieldmatrix.shape[2]))
leadfieldmatrix = leadfieldmatrix
leadfieldmatrix.shape

def matchingpursuit(leadfieldmatrix, potential):
    corrcoef = []
    M, N = 75, 245331
    assert leadfieldmatrix.shape[0] == M
    assert leadfieldmatrix.shape[1] == 3 * N
    assert potential.shape[0] == M 

    for i in range(0, leadfieldmatrix.shape[1]):
        corr_num = np.inner(potential[:], leadfieldmatrix[:, i])
        corr_den = np.linalg.norm(potential) * np.linalg.norm(leadfieldmatrix[:, i])
        corrcoef.append(corr_num / corr_den)
    sources = np.asarray(corrcoef)
    soruces = np.argmax(sources)
    return sources

potential = np.cov(leadfieldmatrix)[:, 0]
sources = matchingpursuit(leadfieldmatrix, potential)
estimated = []
for idx in range(leadfieldmatrix.shape[1]):
    estimated.append(np.argmax(leadfieldmatrix[:, idx] - sources[idx, ]))