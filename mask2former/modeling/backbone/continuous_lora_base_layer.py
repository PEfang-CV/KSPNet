from typing import Dict, Any
import math
import torch
from  torch import nn, Tensor

class ContinuousLoRALayer():
    def __init__(
        self, 
        d: int, 
        k: int, 
        r: int, 
        lora_alpha: int, 
        scaling: float,
        lora_dropout: float,
        merge_weights: bool,
        number_of_tasks: int,
        current_task: int,
    ):

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.merged = False
        self.merge_weights = merge_weights

        assert isinstance(number_of_tasks, int)
        assert number_of_tasks > 0
        self.number_of_tasks = number_of_tasks
        self.current_task = current_task

        self.lora_adapters = nn.ModuleDict({
            str(task_id): nn.ParameterDict({
                'lora_A': nn.Parameter(Tensor(r, d).zero_(), requires_grad=False),
                'lora_B': nn.Parameter(Tensor(k, r).zero_(), requires_grad=False)
            }) for task_id in range(self.number_of_tasks)
        })
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = scaling

    def reset_parameters(self):
        for task_params in self.lora_adapters.values():
            nn.init.kaiming_uniform_(task_params['lora_A'], a=math.sqrt(5))  
            nn.init.zeros_(task_params['lora_B']) 

