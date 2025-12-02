import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoConfig

class BERTTeacher(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        print(f"Loading Teacher Model: {model_name}....")

        self.config = AutoConfig.from_pretrained(model_name) 
        self.config.output_hidden_states = True
        self.config.output_attentions = True 

        self.model = AutoModelForMaskedLM.from_pretrained(model_name, config=self.config)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outuputs = self.model(
            input_ids=input_ids,        
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outuputs
    