

def get_all_text(SAVE_DATA,config,token_budget):
    save_token_budget = SAVE_DATA[token_budget]
    texts = []
    sample_idxs = list(range(config.num_samples))
    completion_idxs = list(range(config.num_completions))
    for sample_idx in sample_idxs:
        for completion_idx in completion_idxs:
            texts.append(save_token_budget[sample_idx]['completions'][completion_idx]['text'])
    return texts