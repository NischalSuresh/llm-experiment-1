from transformers import AutoConfig, AutoModelForCausalLM

def get_model(model_name="HuggingFaceTB/SmolLM2-135M", from_scratch=False):
    """
    Initialize a HF model
    If from_scratch is True, then the model is initialized with random weights
    """
    if from_scratch:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
    
    else:
        model - AutoModelForCausalLM.from_pretrained(model_name)
        
    return model

if __name__ == "__main__":
    test_model = get_model(from_scratch=True)
    print(f"Model initialized and num of params = {test_model.num_parameters():,}")