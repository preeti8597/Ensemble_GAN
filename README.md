# Ensemble_GAN
It is an extended GAN approach to overcome the GAN pathologies using explicit ensembling approach. 
To make GAN more robust and efficient an ensemble approach is employed in discriminator side using 1 generator-2 discriminator ensemble GAN model .
The ensemble is developed using goodfellow approach as the base GAN model that works in terms of simplified loss function found aimed with minimizing distance between the generated distribution and the original distribution.
This ensembles obtain  goal of minimizing GAN pathologies by training multiple neural networks without incurring additional cost or to keep the additional cost as minimum as possible. Here, the training time of an ensemble is same as the training time of a single model.
For evaluation of model we have chosen  Frechet Inception Distance (FID), Mode Collapse Evaluation, SSIM, Total time computed and  Inception Score (IS) as the quantitative parameters to determine the nature of working of proposed model.
Proposed model work incredibly well specially for small datasets which is considered as a desirable characteristics for deep learning techniques. 
