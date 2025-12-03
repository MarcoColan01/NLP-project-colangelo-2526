import torch 
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

class DataLoader:
    def __init__(self, tokenizer, max_length=128, batch_size=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size    

        self.keywords = [
            "history", "empire", "war", "ancient", "king", "queen", # Storia
            "geography", "river", "mountain", "capital", "population", # Geografia
        ]
    
    # This function loads and preprocesses the dataset
    def filter_by_keywords(self, example):
        text = example['text'].lower()
        return any(keyword in text for keyword in self.keywords)

    #This function tokenizes the dataset
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_special_tokens_mask=True
        )

    # This function load the dataset, filter it and tokenize it
    def get_dataset(self, split="train", num_samples=10000):
        print(f" Loading Wikipedia ({split}) in streaming mode...")

        # Load the Wikipedia en dataset in streaming mode
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split, streaming=True)

        # Filter the dataset based on arguments
        print("Filtering the dataset...")
        dataset = dataset.filter(self.filter_by_keywords)

        #Take the first num_samples samples
        dataset = dataset.take(num_samples)

        #Apply the tokenization function to the dataset
        print("Tokenizing the dataset...")
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True).remove_columns(["text"])

        return tokenized_dataset
    
    def get_dataloader(self, split="train", num_samples=10000):
        dataset = self.get_dataset(split, num_samples)

        # Create DataLoader with DataCollatorForLanguageModeling for MLM (Masked Language Modeling)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True,               # Enable Masked Language Modeling
            mlm_probability=0.15    # 15% of the tokens will be masked
        )

        # Create DataLoader with PyTorch  
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=data_collator
        )

        return dataloader