## Basic Pytorch implementation of the PCGrad

This is a basic implementation of the PC_GRAD loss suggested in paper [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf) 
Please see here for the official implementation and citation info [PC_Grad](https://github.com/tianheyu927/PCGrad).

## Note: 
1) The loss function converges for me on a different project.
2) Please don't forget to call the ```optimizer.step()``` after calculating gradients from the loss function.

## Usage:

```

output = net(input)
loss1 = criterion_one(output[0], labels[0])
loss2 = criterion_two(output[1], labels[1])

# Optimizer 
PCGrad_loss([loss1, loss2], [optimizer], [net], device):

optimizer.step()
```

## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}
```
