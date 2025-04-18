import torch

class InterpretableCNN(torch.nn.Module):    
    
    def __init__(self, classes=2, sampleChannel=30, sampleLength=384 ,N1=16, d=2,kernelLength=64):
        super(InterpretableCNN, self).__init__()
        self.pointwise = torch.nn.Conv2d(1,N1,(sampleChannel,1))
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1) 
        self.activ=torch.nn.ReLU()       
        self.batchnorm = torch.nn.BatchNorm2d(d*N1,track_running_stats=False)       
        self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))         
        self.fc = torch.nn.Linear(d*N1, classes)        
        self.softmax=torch.nn.LogSoftmax(dim=1)

    def forward(self, inputdata):
        intermediate = self.pointwise(inputdata)        
        intermediate = self.depthwise(intermediate) 
        intermediate = self.activ(intermediate) 
        intermediate = self.batchnorm(intermediate)          
        intermediate = self.GAP(intermediate)     
        intermediate = intermediate.view(intermediate.size()[0], -1) 
        intermediate = self.fc(intermediate)    
        output = self.softmax(intermediate)   

        return output  
