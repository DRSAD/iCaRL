# PyTorch Implementation of  iCaRL



A PyTorch Implementation of [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725).



## requirement

python3.6

Pytorch1.3.0 linux

numpy

PIL



## run

```shell
python -u main.py
```





# Result

Resnet18+CIFAR100



| incremental step    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9|
| ------------------- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| iCaRL test accuracy | 83 |74|69|64|61|58|56|54|52|50|
| hybrid1 | 83 | 72 | 66 | 60 | 57 | 53 | 51 | 47 | 45 | 42 |

