
from .continuous_lora_base_layer import ContinuousLoRALayer
import torch
from  torch import nn, Tensor
import torch.nn.functional as F

from .conv2d import Conv2d

from detectron2.utils.comm import get_local_rank

class ContinuousConvLoRALayer(nn.Module, ContinuousLoRALayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            number_of_tasks: int,
            current_task:int,
            conv_module: Conv2d = Conv2d,
            r: int = 0,   
            lora_alpha: int = 1, 
            lora_dropout: float=0.,
            merge_weights: bool = True, 
            **kwargs
        ):
        self.parents_initialized = False
        super(ContinuousConvLoRALayer, self).__init__()
        
        self.conv: Conv2d = conv_module(in_channels, out_channels, kernel_size, **kwargs)

        number_of_tasks=number_of_tasks[0]
        self.r=r
        lora_alpha=lora_alpha[0]
        lora_dropout=lora_dropout[0]
        merge_weights=merge_weights[0]
        self.merge_weights=merge_weights
        self.open=True

        r_ = r * kernel_size
        d_ = in_channels * kernel_size
        k_ = out_channels//self.conv.groups*kernel_size

        if r>0:
            ContinuousLoRALayer.__init__(self, d=d_, k=k_, r=r_, scaling=lora_alpha/r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, number_of_tasks=number_of_tasks,current_task=current_task)
            self.conv.weight.requires_grad = False 
            assert isinstance(kernel_size, int)
            assert isinstance(r, int)
            assert r > 0

        self.parents_initialized = True
        self.reset_parameters()
        self.merged = False

        self.device = torch.device(f"cuda:{get_local_rank()}")
        
    
        self.total_increment = torch.zeros_like(self.conv.weight).to(self.device)   
        self.current_task=current_task
        for task in range(1, self.current_task):
            self.total_increment= self.total_increment+ (self.lora_adapters[str(task)]['lora_B'] @ self.lora_adapters[str(task)]['lora_A']).view(self.conv.weight.shape).to(self.device) 
        self.F_init = (torch.abs(self.conv.weight) ** 2).to(self.device)  

    def count_trainable_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def reset_parameters(self):
        if self.parents_initialized:
            self.conv.reset_parameters()
        

    def forward(self, x: Tensor) -> Tensor:

        if self.r > 0 and not self.merged:  
            all_increment=self.total_increment+ (self.lora_adapters[str(self.current_task)]['lora_B'] @ self.lora_adapters[str(self.current_task)]['lora_A']).view(self.conv.weight.shape)           
            initer_task=torch.abs(self.total_increment) * (self.lora_adapters[str(self.current_task)]['lora_B'] @ self.lora_adapters[str(self.current_task)]['lora_A']).view(self.conv.weight.shape)
            loss_cis =0.5 * torch.sum(self.F_init * (all_increment ** 2))+torch.sqrt(torch.sum(initer_task ** 2)+ 1e-8)
           
            outputShort=self.conv._conv_forward(
                x, 
                self.conv.weight+self.total_increment* self.scaling+ (self.lora_adapters[str(self.current_task)]['lora_B'] @ self.lora_adapters[str(self.current_task)]['lora_A']).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )

            if self.conv.norm is not None:
                outputShort = self.conv.norm(outputShort)
            if self.conv.activation is not None:
                outputShort = self.conv.activation(outputShort)
            return outputShort, loss_cis

            
        return self.conv(x),torch.tensor(0).to(self.device)

