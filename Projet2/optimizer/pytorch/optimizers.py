from abc import abstractmethod
import torch

class optimizer(object):
    @abstractmethod
    def compute_update(self, gradients):
        raise NotImplementedError('compute_update function is not implemented.')

class SGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate

    def update(self, model):
        for p in model.parameters():
            p.grad *= self.learning_rate

class MomentumSGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.m = None

    def update(self, model):
        if self.m is None:
            self.m = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            self.m[i] = self.rho * self.m[i] + p.grad
            p.grad = self.learning_rate * self.m[i]


class RMSPropOptimizer(optimizer):
    def __init__(self, args):
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.r = None
        self.delta = args.delta
        

        

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
           
           self.r[i] = self.tau * self.r[i] + ( 1 - self.tau ) * torch.multiply( p.grad, p.grad ) 
           
           p.grad *= self.learning_rate/(self.delta + torch.sqrt(self.r[i]))
        
          


class AdamOptimizer(optimizer):
    def __init__(self, args):
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.learning_rate = args.learning_rate
        self.delta = args.delta
        self.iteration = None
        self.m1 = None
        self.m2 = None
        self.m1_corrected = None
        self.m2_corrected = None

    def update(self, model):
        if self.m1 is None:
            self.m1 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.m2 is None:
            self.m2 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(model.parameters()):
            
            self.m1[i] = self.beta1 * self.m1[i] + ( 1 - self.beta1 ) * p.grad
            self.m2[i] = self.beta2 * self.m2[i] + ( 1 - self.beta2 ) * torch.multiply( p.grad, p.grad )
            
            #corrected bias
            
            self.m1_corrected = self.m1[i] / ( 1 - self.beta1**self.iteration)
            self.m2_corrected = self.m2[i] / ( 1 - self.beta2**self.iteration)
            
            #update grad
            p.grad = self.learning_rate * self.m1_corrected/(self.delta + torch.sqrt(self.m2_corrected))
            

        self.iteration = self.iteration + 1
        

def createOptimizer(args):
    if args.optimizer == "sgd":
        return SGDOptimizer(args)
    elif args.optimizer == "momentumsgd":
        return MomentumSGDOptimizer(args)
    elif args.optimizer == "rmsprop":
        return RMSPropOptimizer(args)
    elif args.optimizer == "adam":
        return AdamOptimizer(args)
