# SDENet-UQ-ESL
## SDENet as Uncertainty Quantification Method for EEG Source Localization

UPDATE: [Paper Under Peer-Review]

EEG source localization remains a challenging problem given the uncertain conductivity values of the volume conductor models (VCMs). As uncertain conductivities vary across people, they may considerably impact the forward and inverse solutions of the EEG, leading to an increase in localization mistakes and misdiagnoses of brain disorders. Calibration of conductivity values using uncertainty quantification (UQ) techniques is a promising approach to reduce localization errors. The widely-known UQ methods involve Bayesian approaches, which utilize prior conductivity values to derive their posterior inference and estimate their optimal calibration. However, these approaches have two significant drawbacks: solving for posterior inference is intractable, and choosing inappropriate priors may lead to increased localization mistakes. 

<img src="figures/Figure 1.png" width="708"/>

This study used the Neural Stochastic Differential Equations Network (SDE-Net), a combination of dynamical systems and deep learning techniques that utilizes the Wiener process to minimize conductivity uncertainties in the VCM and improve the inverse problem. Results revealed that SDE-Net generated a lower localization error rate in the inverse problem compared to Bayesian techniques. Future studies may employ new stochastic dynamical systems-based techniques as a UQ technique to address further uncertainties in the EEG Source Localization problem. 


<img src="figures/Figure 2.png" width="708"/>

## EEG Forward Problem
The forward problem is mathematically expressed as $\Phi = LJ(r)$ where it requires to calculate the electrical potentials $\Phi$ $(V)$, lead field matrix $L$ $(V/m)$, and the current density $J$ $(A/m^2)$ located at the source position $r$ ($mm$)

<img src="figures/Figure 3.gif" width="708"/>

## EEG Inverse Problem
The inverse problem used the Matching Pursuit Algorithm (MPA) to compute estimated source distribution.

<img src="figures/Figure 4.JPG" width="708"/>
