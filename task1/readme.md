@[toc](PyTorch的基本概念)
# 1.什么是Pytorch，为什么选择Pytroch？
- 作为NumPy的替代品，可以利用GPU的性能进行计算
- 作为一个高灵活性、速度快的深度学习平台
- Pytorch相比Tensorflow而言，它的设计初衷是简单易用用，所以它是基于动态图进行实现的，从而方便调试。当然，Tensorflow在1.5版的时候就引入了Eager Execution机制实现了动态图，但它还是默认使用静态图。
# 2.Pytroch的安装
* 两个方式 conda 和 pip
* https://pytorch.org/get-started/locally/
* 查看系统软件版本兼容关系【使用conda安装关系不大】 https://github.com/pytorch/pytorch 
## window安装
* 配置Python环境：
* 安装anaconda，**创建虚拟环境**  https://blog.csdn.net/leviopku/article/details/84548816
```angular2html
#创建conda虚拟环境
conda create -n conda36 python=3.6
删除环境（不要乱删啊啊啊）
conda remove -n py36 --all
jupyter中添加conda虚拟环境

https://blog.csdn.net/u014665013/article/details/81084604

https://www.jianshu.com/p/0432155d1bef

'''
python -m ipyhandson-ml install --user --name handson-ml --display-name "handson-ml"
'''

#安装 pytorch
conda install pytorch-cpu torchvision-cpu -c pytorch
#验证
>>> import torch
>>> import torchvision
```

# gpu 版本安装 

### 问题
* pycharm中导入torch的conda的虚拟环境，但是只有torch的相关包，anaconda默认的包不可以用
# 2.Pytroch的安装【方法二，针对以上报错】，pip安装
```angular2html
# Python 3.6
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl
pip3 install torchvision

```
## jupyter notebook 安装
https://www.brothereye.cn/python/335/

## linux unbantu18版安装
```angular2html
pip3 install torch torchvision
```

# 3.PyTorch基础概念 
参考
https://blog.csdn.net/herosunly/article/details/88892326
https://blog.csdn.net/herosunly/article/details/88915673
# 4.Pytorch基本实现
```

# 4.PyTorch
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
```
参考
https://blog.csdn.net/herosunly/article/details/89036914