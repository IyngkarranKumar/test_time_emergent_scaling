#macros - maybe better way to do this

TOKENIZER_NAMES = [
    "EleutherAI/pythia-70m",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/QwQ-32B", 
    "microsoft/Phi-4-reasoning",
    "open-thoughts/OpenThinker-32B"
]

qwen_end_of_input_ids = [151644, 77091, 198]  # '<|im_start|>assistant\n'
gemma_end_of_input_ids = [106, 2516, 108]  # '<start_of_turn>model\n'
deepseek_end_of_input_ids = [151645, 151648, 198] #'<｜Assistant｜><think>\n'
qwq_end_of_input_ids = [151645, 198, 151644, 77091, 198] #<|im_end|>\n<|im_start|>assistant\n
phi_end_of_input_ids = [100265,100264,78191,100266] #<|im_end|><|im_start|>assistant<|im_sep|>

qwen25_pad_token, qwen25_pad_token_id = "!",0
deepseek_pad_token,deepseek_pad_token_id = "!",0
qwq_pad_token,qwq_pad_token_id = "<|endoftext|>",151643
phi_pad_token,phi_pad_token_id = "<|dummy_85|>",100349