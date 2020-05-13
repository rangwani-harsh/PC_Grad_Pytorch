## Basic Pytorch implementation of the PCGrad

This is a basic implementation of the PC_GRAD loss suggested in paper [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf) 
Please see here for the official implementation and citation info [PC_Grad](https://github.com/tianheyu927/PCGrad).

Note: 
1) The loss function converges for me on a different project.
2) Please don't forget to call the ```optimizer.step()``` after calculating gradients from the loss function.


