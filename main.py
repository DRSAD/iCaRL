from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import torch

numclass=10
feature_extractor=resnet18_cbam()
img_size=32
batch_size=128
task_size=10
memory_size=2000
epochs=70
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,img_size,batch_size,task_size,memory_size,epochs,learning_rate)

for i in range(10):
    model.beforeTrain()
    accuracy=model.train()
    KNN_accuracy=model.afterTrain()
    print("NMS准确率："+str(KNN_accuracy.item()))
    torch.save(model.model.state_dict(),'model/accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10))
