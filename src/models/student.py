import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM 

class TinyBERTStudent(nn.Module):
    def __init__(self, teacher_config, student_type = "tiny"):
        super().__init__()

        # TinyBERT structure from the Paper 3
        if student_type == "tiny":
            hidden_size = 312
            num_hidden_layers = 4
            intermmediate_size = 1200
        else :
            raise ValueError("Unsupported student type")
        
        print("Building Student Model....")
        
        # Define student configuration based on attention loss
        self.config = BertConfig(
            vocab_size = teacher_config.vocab_size,
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,  
            num_attention_heads = teacher_config.num_attention_heads,
            intermediate_size = intermmediate_size, 
            max_position_embeddings = teacher_config.max_position_embeddings,
            output_hidden_states = True,
            output_attentions = True
        )

        # Initialize the student model and a linear layer to match dimensions
        self.model = BertForMaskedLM(self.config)
        self.fit_dense = nn.Linear(hidden_size, teacher_config.hidden_size)

        # The forward method serves to pass inputs through the student model
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,        
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs