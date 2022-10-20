# SDENet-UQ-ESL
## SDENet as Uncertainty Quantification Method for EEG Source Localization

EEG source localization remains a challenging problem given the uncertain conductivity values of the volume conductor models (VCMs). As uncertain conductivities vary across people, they may considerably impact the forward and inverse solutions of the EEG, leading to an increase in localization mistakes and misdiagnoses of brain disorders. Calibration of conductivity values using uncertainty quantification (UQ) techniques is a promising approach to reduce localization errors. The widely-known UQ methods involve Bayesian approaches, which utilize prior conductivity values to derive their posterior inference and estimate their optimal calibration. However, these approaches have two significant drawbacks: solving for posterior inference is intractable, and choosing inappropriate priors may lead to increased localization mistakes. 

<img src="figures/Figure 1.png" width="708"/>

$$ \mathbf{X}{t+1} = f(\mathbf{X}_t, t)\text{d}t + g(\mathbf{X}_t, t)\sqrt{\Delta t}\mathbf{W}_t $$
where $\mathbf{W}_k$ is the standard Gaussian random variable. The objective function for training SDE-Net is expressed as 
\begin{equation}
\text{min}_{\theta_f}\mathbb{E}_{x_0}\mathbb{E}\left(L \left(\mathbf{X_T}\right)\right) + \text{max}_{\theta_g}\mathbb{E}_{x_0}g\left(x_0; \theta_g\right)  
\end{equation} 

This study used the Neural Stochastic Differential Equations Network (SDE-Net), a combination of dynamical systems and deep learning techniques that utilizes the Wiener process to minimize conductivity uncertainties in the VCM and improve the inverse problem. Results revealed that SDE-Net generated a lower localization error rate in the inverse problem compared to Bayesian techniques. Future studies may employ new stochastic dynamical systems-based techniques as a UQ technique to address further uncertainties in the EEG Source Localization problem. 
D:\dsai-thesis\results\potentials

<img src="figures/Figure 2.png" width="708"/>

## EEG Forward Problem
<img src="figures/Figure 3.gif" width="708"/>
