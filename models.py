#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from loss import cuda


class Net(nn.Module):
    def __init__(self, latent, num_classes=100, norm=True, scale=True):
        super(Net,self).__init__()
        self.extractor = Extractor()
        self.embedding = Embedding(latent)
        self.proj = Projection(latent)
        self.classifier = Classifier(latent,num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale
        self.latent = int(latent)

   
    def norm_emb(self, x):
        if self.norm:
            x = self.l2_norm(x)
        return x

    def scale_(self, x):    
        if self.scale:
            x = self.s * x
        return x


    def vib(self, feature):
        mu = feature[:,:int(self.latent/2)]
        mu = self.norm_emb(mu)
        std = F.softplus(feature[:,int(self.latent/2):]-5,beta=1)

        return mu, std, self.reparametrize_n(mu,std,1)


    def _forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x1_,x1 = self.proj(x)
      
        return x1_,x1

    
    def forward(self, x):
        x,_ = self._forward(x)
        
        mu ,std, encoding = self.vib(x)
        mu = self.scale_(mu)

        encoding = self.scale_(self.l2_norm(encoding))

        logit = self.classifier(encoding)

        return logit, mu, std, encoding


    def helper_extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x1_,x1 = self.proj(x)

        x1_,_,_ = self.vib(x1_)
        x1_ = self.l2_norm(x1_)

        x1,_,_ = self.vib(x1)
        x1 = self.l2_norm(x1)

        return x1_,x1
    
    def forward_wi_fc1(self, x):
        _,x = self.helper_extract(x)
        logit = self.classifier(x)
        
        return logit, x

    def forward_wi_fc1_(self, x):
        x1_,_ = self.helper_extract(x)
        logit_ = self.classifier(x1_)
        
        return logit_, x1_
    

    def extract(self, x):
        x = self.helper_extract(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))



    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
        

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

    def xavier_init(ms):
      for m in ms :
          if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
              nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
              m.bias.data.zero_()

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor,self).__init__()
        basenet = models.resnet50(pretrained=True)

        self.extractor = nn.Sequential(*list(basenet.children())[:-1])
        #self.dropout = nn.Dropout(p=0.2)
        

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        return x


class Embedding(nn.Module):
    def __init__(self,latent):
        super(Embedding,self).__init__()
        self.fc = nn.Linear(2048, 512)
        #self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc(x)
       # x = self.relu(x)
        
        return  x


class Projection(nn.Module):
    def __init__(self,latent):
        super(Projection,self).__init__()
        self.fc = nn.Linear(512, latent) 
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.2)
        
    
    def l2_norm1(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.fc(x)
        fc1_ = self.l2_norm1(x)
        #x1_ = self.relu(x)
        
        return fc1_, x


class Classifier(nn.Module):
    def __init__(self, latent,num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(int(latent/2), num_classes, bias=False)
        
    def forward(self, x):
        x = self.fc(x)    
       
        return x
