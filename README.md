# AIHacks4Good2022-BiasBuster
## About The Project
This is an unfinished entry to the EthicalAI Hackathon 2022. 

We are looking at providing a tool that helps identify biases from doctors. Not all doctors will have the same training, nor the same set of questions they ask when a patient approaches them, and this can sometimes perpetuate a bias. 

Furthermore, certain biases will create statistical features that indicate their presence. Using machine learning to attempt to identify these features should be more advantageous than a mere statistical analysis, since the model can attempt to pick up on more abstract features.

For this purpose, we have trained a discriminator as part of a GAN. This discriminator is trained on data gathered from a 'gold-standard doctor'. This discriminator can be used as a tool on future data from doctors to indicate whether they might have systematic biases.
