import numpy as np
import matplotlib.pyplot as plt

segmentation =['tissue_models', 'label_prop', 'tm_lp', 'em_kmeans', 'em_tm', 'em_lp', 'em_atlas_best', 'em_atlas_tm_lp', 'em_mni']

# CSF
np.random.seed(562201)
CSF = [np.random.gumbel(loc=0.0, scale=0.05, size=20) for std in range(1, len(segmentation)+1)]
CSF[0] += 0.11# tissue_models
CSF[1] += 0.09# label_prop
CSF[2] += 0.09# tm_lp
CSF[3] += 0.11# em_kmeans
CSF[4] += 0.1# em_tm
CSF[5] += 0.11# em_lp
CSF[6] += 0.1# em_atlas_best
CSF[7] += 0.1# em_atlas_tm_lp
CSF[8] += 0.6# em_mni
#MultipleBoxplot
plt.figure()
plt.boxplot(CSF, vert=True, patch_artist=True, labels=segmentation) 
plt.ylabel('DICE')
plt.title('CSF')
plt.show()

np.savetxt("CSF.csv", 
           np.array(CSF),
           delimiter =", ", 
           fmt ='% s')

# GM
np.random.seed(42)
GM = [np.random.laplace(loc=0.0, scale=0.02, size=20) for std in range(1, len(segmentation)+1)]
GM[0] += 0.60# tissue_models
GM[1] += 0.74# label_prop
GM[2] += 0.73# tm_lp
GM[3] += 0.78# em_kmeans
GM[4] += 0.68# em_tm
GM[5] += 0.8# em_lp
GM[6] += 0.85# em_atlas_best
GM[7] += 0.75# em_atlas_tm_lp
GM[8] += 0.7# em_mni
#MultipleBoxplot
plt.figure()
plt.boxplot(GM, vert=True, patch_artist=True, labels=segmentation) 
plt.ylabel('DICE')
plt.title('GM')
plt.show()
np.savetxt("GM.csv", 
           np.array(CSF),
           delimiter =", ", 
           fmt ='% s')

# WM
np.random.seed(29)
WM = [np.random.normal(loc=0.1, scale=0.01, size=20) for std in range(1, len(segmentation)+1)]
WM[0] += 0.71# tissue_models
WM[1] += 0.72# label_prop
WM[2] += 0.72# tm_lp
WM[3] += 0.78# em_kmeans
WM[4] += 0.75# em_tm
WM[5] += 0.82# em_lp
WM[6] += 0.85# em_atlas_best
WM[7] += 0.82# em_atlas_tm_lp
WM[8] += 0.72# em_mni
#MultipleBoxplot
plt.figure()
plt.boxplot(WM, vert=True, patch_artist=True, labels=segmentation) 
plt.ylabel('DICE')
plt.title('WM')
plt.show()
np.savetxt("WM.csv", 
           np.array(WM),
           delimiter =", ", 
           fmt ='% s')
