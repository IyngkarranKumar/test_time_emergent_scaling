import torch
import numpy as np
import warnings
import re
import gc
import inspect
import os
if torch.cuda.is_available():
    import pynvml

from math_verify import parse, verify
from scipy.stats import entropy

from torch.nn.functional import softmax, log_softmax

from .workflow_utils import clean_up_gpus
from .profile_utils import gpu_utilisation, gpu_util_manager, top_gpu_tensors

def map_to_3_digits(num):
        """
        Map a number (as int or str) to a 3-digit string, e.g.:
        '70' -> '070', '1' -> '001', '123' -> '123'
        """
        num_str = str(num).strip()
        # Remove any leading zeros, then pad to 3 digits
        try:
            n = int(num_str)
        except Exception:
            return num_str  # fallback: return as is if not a number
        return f"{n:03d}"

def calculate_entropy_via_weighted_surprisal(dist):

    if not isinstance(dist, np.ndarray):
        dist = np.array(dist)

    safe_dist = dist + 1e-10 #for entropy calculation
    norm_dist = safe_dist / sum(safe_dist) #normalise 

    weighted_surprisal = -1*norm_dist * np.log2(norm_dist)
    weighted_surprisal = np.where(np.isinf(weighted_surprisal) | np.isneginf(weighted_surprisal) | np.isnan(weighted_surprisal), 0, weighted_surprisal) #fliter our problematic values 

    entropy = np.sum(weighted_surprisal)

    return entropy


#scoring utils 
def score_answer(model_answer: str, answer_string: str,config) -> int:

    #for gpqa
    if config.dataset_name=="Idavidrein/gpqa":
        normalized_model = model_answer.strip().upper() if model_answer else ""
        normalized_answer = answer_string.strip().upper() if answer_string else ""
        score = 1 if normalized_model == normalized_answer else 0
    else:
        gold = parse(answer_string)
        model_answer_parsed = parse(model_answer)
        score_math_verify = verify(gold, model_answer_parsed) #gold must come before score

        # Also do an exact string match - and if either of these is 1 set that as score
        score_exact_string_match = 0
        if model_answer is not None and answer_string is not None:
            model_answer_3_digits = map_to_3_digits(model_answer.strip())
            answer_string_3_digits = map_to_3_digits(answer_string.strip())
            if model_answer_3_digits == answer_string_3_digits:
                score_exact_string_match = 1
            else:
                score_exact_string_match = 0

        score = max(score_math_verify, score_exact_string_match) #take the max of the two scores

    return 1 if score else 0


def calculate_answer_probability(answer_tokens, reasoning_trace, force_end, model, tokenizer, config, first_token_logits,return_entropy=False):
    
    """
    Calculates probability for arbitrary answer sequence
    return_entropy: if True, returns entropy of over vocab N grams, where N is answer token length

    [tokenizer.decode(d.argmax()) for d in distribution_store] #useful for debugging (need interact first)
    """
    if config.aime_answer_prompt:
        answer = tokenizer.decode(answer_tokens,skip_special_tokens=True)
        answer = map_to_3_digits(answer)
        answer_tokens = tokenizer(answer).input_ids
    

    probs = []
    distribution_store = []

    #first token prob 
    answer_probs = softmax(first_token_logits,dim=-1)
    distribution_store.append(answer_probs)
    first_token_prob = answer_probs[answer_tokens[0]].item() #check 
    probs.append(first_token_prob)

    #prepare batched inputs for remaining tokens
    if len(answer_tokens) > 1:
        input_texts = []
        for tok_idx in range(1,len(answer_tokens)):
            input_texts.append(reasoning_trace + ' ' + force_end + tokenizer.decode(answer_tokens[:tok_idx],skip_special_tokens=True)) #list of strings, before going to padded ids

        device = next(model.parameters()).device #get device that model is on
        batched_inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(device)

        #generate logits
        with torch.no_grad():
            outputs = model(**batched_inputs) #just use forward pass to get logits
            #outputs.logits is [batch_size, seq_len, vocab_size]
            last_token_logits = outputs.logits[:,-1,:] #logits used to predict next token 
               
        #process logits 
        for tok_idx, tok_id in enumerate(answer_tokens[1:]):
            answer_logits = last_token_logits[tok_idx]
            answer_probs = softmax(answer_logits, dim=-1)
            distribution_store.append(answer_probs)
            probs.append(answer_probs[tok_id].item())

        if not config.DEBUG_MODE:
            del batched_inputs, outputs #remove tensors that take up memory on device
            gc.collect()
            clean_up_gpus()

    answer_prob = np.prod([p for p in probs]) if np.any(probs) else 0
    if isinstance(answer_prob, torch.Tensor) or isinstance(answer_prob, np.ndarray):
        answer_prob = answer_prob.item()

    if return_entropy:
        entropy_sum = 0
        for dist in distribution_store:
            safe_dist = (dist + 1e-10).cpu().numpy()
            safe_dist = safe_dist / np.sum(safe_dist)

            
            # Handle potential numerical issues
            safe_dist = np.clip(safe_dist, 1e-10, 1.0)
            log_probs = np.log2(safe_dist)

            #calculate weighted surprisals, filter out nans/infs
            weighted_suprisal = -1*safe_dist * log_probs
            filter_condition = np.isnan(weighted_suprisal) | np.isinf(weighted_suprisal) | np.isneginf(weighted_suprisal)
            weighted_suprisal = np.where(filter_condition, 0.0, weighted_suprisal)

            entropy = np.sum(weighted_suprisal)
            
            entropy_sum += entropy

        avg_entropy = entropy_sum / len(answer_tokens)

        return answer_prob, avg_entropy

    else:
        return answer_prob 


def batch_calculate_answer_probability(answer_tokens, reasoning_traces, force_end_delim, model, tokenizer, config, first_token_logits, return_entropy=False):
    """
    calculate answer probability for a batch of inputs 

    returns: probs, entropies
    probs: list of probabitlies of ground truth answer tokens
    entropies: list of entropies of ground truth answer tokens (sum of entropy over N_G tokens)

    [tokenizer.decode(d.argmax()) for d in distribution_store] #useful for debugging (need interact first)
    """

    batch_size = len(answer_tokens)
    max_answer_len = max(len(tokens) for tokens in answer_tokens)
    individual_probs = [[] for _ in range(batch_size)]
    individual_entropies = [[] for _ in range(batch_size)]

    # First token probs for all samples
    first_token_prob_dist = softmax(first_token_logits[0], dim=-1) #first token logits are all the same 
    safe_dist = (first_token_prob_dist+1e-10).cpu().numpy()
    safe_dist = safe_dist/sum(safe_dist)

    #our filtered weighted surprisal seems more robust to numerical issues
    weighted_surprisal = -1*safe_dist*np.log2(safe_dist)
    filtered_weighted_surprisal = np.where(np.isnan(weighted_surprisal) | np.isinf(weighted_surprisal) | np.isneginf(weighted_surprisal), 0.0, weighted_surprisal)
    entropy = np.sum(filtered_weighted_surprisal)

    for sample_idx in range(batch_size):
        first_token_prob = first_token_prob_dist[answer_tokens[sample_idx][0]].item()
        individual_probs[sample_idx].append(first_token_prob)
        individual_entropies[sample_idx].append(entropy.item())


    # Process remaining tokens in batches
    for tok_idx in range(1, max_answer_len):
        # Create input texts for samples that have this token position
        input_texts = []
        sample_mapping = [] # Maps batch position to original sample index
        
        for sample_idx in range(batch_size):
            if tok_idx < len(answer_tokens[sample_idx]):
                input_text = reasoning_traces[sample_idx] + ' ' + force_end_delim + tokenizer.decode(answer_tokens[sample_idx][:tok_idx], skip_special_tokens=True)
                input_texts.append(input_text)
                sample_mapping.append(sample_idx)

        if not input_texts:
            continue


        # Prepare batch inputs
        device = next(model.parameters()).device
        batched_inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**batched_inputs)
            pred_logits = outputs.logits[:, -1, :] 

            if not config.DEBUG_MODE:
                del outputs, batched_inputs
                gc.collect()
                clean_up_gpus()

        # Process logits for each sample
        for batch_idx, sample_idx in enumerate(sample_mapping):
            logits = pred_logits[batch_idx]
            answer_probs = softmax(logits, dim=-1)
            next_token = answer_tokens[sample_idx][tok_idx]
            token_prob = answer_probs[next_token].item()
            individual_probs[sample_idx].append(token_prob)
            
            # Calculate entropy
            if return_entropy:
                safe_dist = (answer_probs + 1e-10).cpu().numpy()
                safe_dist = safe_dist / np.sum(safe_dist)

                # Handle potential numerical issues
                safe_dist = np.clip(safe_dist, 1e-10, 1.0)
                log_probs = np.log2(safe_dist)

                #calculate weighted surprisals, filter out nans/infs
                weighted_suprisal = -1*safe_dist * log_probs
                filter_condition = np.isnan(weighted_suprisal) | np.isinf(weighted_suprisal) | np.isneginf(weighted_suprisal) #remove buggy values
                weighted_suprisal = np.where(filter_condition, 0.0, weighted_suprisal)

                entropy = np.sum(weighted_suprisal)

                individual_entropies[sample_idx].append(entropy.item())


    # Calculate final probabilities and average entropies
    final_probs = []
    final_entropies = []
    for sample_idx in range(batch_size):
        ground_truth_tokens = answer_tokens[sample_idx]
        prob = np.prod(individual_probs[sample_idx])
        if isinstance(prob, torch.Tensor) or isinstance(prob, np.ndarray):
            prob = prob.item()
        final_probs.append(prob)
        if return_entropy:
            total_entropy = (np.sum(individual_entropies[sample_idx]) if individual_entropies[sample_idx] else 0)/len(ground_truth_tokens) #normalised for token length
            
            final_entropies.append(total_entropy)

    if return_entropy:
        return final_probs, final_entropies
    else:
        return final_probs, None


def calculate_solution_set_probs_aime_with_end_token(reasoning_trace, model, tokenizer, force_end_delim, first_token_logits, config, batch_size=25):

    """
    Calculate AIME probs by leveraging solution symmetry
    """
    reasoning_trace = reasoning_trace + ' ' + force_end_delim

    final_answer_probs = []

    base_10_digits = [str(a) for a in range(10)]
    base_10_digits_end_token = base_10_digits + ["}"]
    base_10_tokens = tokenizer(base_10_digits).input_ids
    end_token_id = tokenizer("}").input_ids[0]
    base_10_tokens_end_token = tokenizer(base_10_digits_end_token).input_ids
    
    first_token_probs = softmax(first_token_logits,dim=-1)
    p_d1 = first_token_probs[torch.tensor(base_10_tokens).flatten()].squeeze().to(model.device) #get probs for 0-9
     

    cond_reasoning_trace_1 = [f"{reasoning_trace}{digit}" for digit in base_10_digits]
    inputs = tokenizer(cond_reasoning_trace_1, return_tensors="pt",padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_logits = outputs.logits[:,-1,:].squeeze(1)
        output_probs = softmax(pred_logits, dim=-1)
    
    p_d2_given_d1 = output_probs[:,torch.tensor(base_10_tokens_end_token).flatten()].squeeze() #get probs for 0-9 


    #compute necessary final answer probs (could also do at the end)
    p_end_given_d1 = p_d2_given_d1[:,-1] #(10,)
    p_d1_end = (p_d1*p_end_given_d1).flatten()
    final_answer_probs.extend([t.cpu().item() for t in p_d1_end])
    
    

    if not config.DEBUG_MODE:
        del inputs, outputs
        gc.collect()
        clean_up_gpus()

    cond_reasoning_trace_2 = [f"{reasoning_trace} {digita}{digitb}" for digita in base_10_digits for digitb in base_10_digits]

    p_d3_given_d2d1 = []
    for i in range(0, len(cond_reasoning_trace_2), batch_size):
        batch = cond_reasoning_trace_2[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_logits = outputs.logits[:,-1,:].squeeze(1)
            output_probs = softmax(pred_logits, dim=-1)
        batch_cond_2_probs = output_probs[:,torch.tensor(base_10_tokens_end_token).flatten()].squeeze() #get probs for 0-9 
        p_d3_given_d2d1.append(batch_cond_2_probs)

        if not config.DEBUG_MODE:
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

    p_d3_given_d2d1 = torch.cat(p_d3_given_d2d1, dim=0)

    #calaulate end token probs 
    p_end_given_d2d1 = p_d3_given_d2d1[:,-1].reshape(10,10) #(100,) - p("}"|d2d1) - #need to check if i,j is what we think (rather than j,i)
    #p(},d2,d1) = p(},d2|d1)p(d1) = p(}|d2d1)*p(d2|d1)*p(d1)
    #p(},d2|d1)
    p_d1d2_end = torch.multiply(p_end_given_d2d1,p_d2_given_d1[:,:-1])*p_d1

    #get probs for 10-99
    final_answer_probs.extend([t.cpu().item() for t in p_d1d2_end.flatten()[10:101]])


    cond_reasoning_trace_3 = [f"{reasoning_trace} {digita}{digitb}{digitc}" for digita in base_10_digits for digitb in base_10_digits_end_token for digitc in base_10_digits_end_token if digita != "0"]

    p_end_given_d3d2d1 = []
    secondary_batch_size = 10
    for i in range(0, len(cond_reasoning_trace_3), secondary_batch_size):
        batch = cond_reasoning_trace_3[i:i + secondary_batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                outputs = model(**inputs)
                pred_logits = outputs.logits[:,-1,:].squeeze(1)
                p_end_given_d3d2d1.append(softmax(pred_logits,dim=-1)[:,end_token_id]) #we only need p(end|d3d2d1); nothing else. list of tensors

        if not config.DEBUG_MODE:
            del inputs, outputs
            gc.collect()
            clean_up_gpus()

    p_end_given_d3d2d1 = torch.cat(p_end_given_d3d2d1, dim=0) #list of tensors into single tensor


    p_d1d2 = p_d2_given_d1[:,:-1] * p_d1 # (10,10). index until -1 because we don't want to include the end token
    p_d1d2d3 = p_d3_given_d2d1[:,:-1] * p_d1d2.flatten().unsqueeze(1) #(100,10)
    p_d1d2d3_end = p_end_given_d3d2d1 * p_d1d2d3.flatten().unsqueeze(1) #(1000,)


    final_answer_probs.extend([t.cpu().item() for t in p_d1d2d3_end.flatten()[100:1001]]) #get probs for 100-999



    return final_answer_probs


def calculate_solution_set_probs_aime_with_aime_prompt(reasoning_trace, model, tokenizer, force_end_delim, first_token_logits, config,batch_size=25,lm_head_manual=True,log_probs=False):
    """
    Calculate AIME probs by leveraging solution symmetry
    Requires aime prompt
    """

    reasoning_trace = reasoning_trace + ' ' + force_end_delim

    final_answer_probs = []
    final_answer_probs_log = []

    base_10_digits = [str(a) for a in range(10)]
    base_10_tokens = tokenizer(base_10_digits, add_special_tokens=False).input_ids

    # Support for log_probs == "both"
    log_probs_mode = log_probs
    if isinstance(log_probs, str) and log_probs.lower() == "both":
        log_probs_mode = "both"
    else:
        log_probs_mode = bool(log_probs)

    # First token probabilities
    if log_probs_mode is True:
        first_token_log_probs = log_softmax(first_token_logits, dim=-1)
        log_p_d1 = first_token_log_probs[torch.tensor(base_10_tokens).flatten()].squeeze().to(model.device)
    elif log_probs_mode == "both":
        first_token_log_probs = log_softmax(first_token_logits, dim=-1)
        first_token_probs = softmax(first_token_logits, dim=-1)
        log_p_d1 = first_token_log_probs[torch.tensor(base_10_tokens).flatten()].squeeze().to(model.device)
        p_d1 = first_token_probs[torch.tensor(base_10_tokens).flatten()].squeeze().to(model.device)
    else:
        first_token_probs = softmax(first_token_logits, dim=-1)
        p_d1 = first_token_probs[torch.tensor(base_10_tokens).flatten()].squeeze().to(model.device)

    cond_reasoning_trace_1 = [f"{reasoning_trace}{digit}" for digit in base_10_digits]
    inputs = tokenizer(cond_reasoning_trace_1, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            if lm_head_manual:
                outputs = model.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                    output_attentions=False,
                )
                last_token_hidden_state = outputs.last_hidden_state[:, -1, :]
                output_log_probs = log_softmax(model.lm_head(last_token_hidden_state), dim=-1)
                output_probs = softmax(model.lm_head(last_token_hidden_state), dim=-1)
            else:
                outputs = model(**inputs)
                pred_logits = outputs.logits[:, -1, :].squeeze(1)
                output_log_probs = log_softmax(pred_logits, dim=-1)
                output_probs = softmax(pred_logits, dim=-1)
            output_log_probs = output_log_probs.to(model.device)
            output_probs = output_probs.to(model.device)

        if not config.DEBUG_MODE:
            try:
                del inputs, outputs
            except Exception as e:
                warnings.warn(f"Error deleting variables: {e}")
            gc.collect()
            clean_up_gpus()

    if log_probs_mode is True:
        log_p_d2_given_d1 = output_log_probs[:, torch.tensor(base_10_tokens).flatten()].squeeze()  # (10,10)
        log_p_d1d2 = log_p_d1.unsqueeze(1) + log_p_d2_given_d1
    elif log_probs_mode == "both":
        log_p_d2_given_d1 = output_log_probs[:, torch.tensor(base_10_tokens).flatten()].squeeze()  # (10,10)
        log_p_d1d2 = log_p_d1.unsqueeze(1) + log_p_d2_given_d1
        p_d2_given_d1 = output_probs[:, torch.tensor(base_10_tokens).flatten()].squeeze()  # (10,10)
        p_d1d2 = p_d2_given_d1 * p_d1.unsqueeze(1)
    else:
        p_d2_given_d1 = output_probs[:, torch.tensor(base_10_tokens).flatten()].squeeze()  # (10,10)
        p_d1d2 = p_d2_given_d1 * p_d1.unsqueeze(1)

    cond_reasoning_trace_2 = [f"{reasoning_trace} {digita}{digitb}" for digita in base_10_digits for digitb in base_10_digits]

    if log_probs_mode is True:
        log_p_d3_given_d2d1 = []
    elif log_probs_mode == "both":
        log_p_d3_given_d2d1 = []
        p_d3_given_d2d1 = []
    else:
        p_d3_given_d2d1 = []

    for i in range(0, len(cond_reasoning_trace_2), batch_size):
        batch = cond_reasoning_trace_2[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                if lm_head_manual:
                    outputs = model.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                        output_attentions=False,
                    )
                    last_token_hidden_state = outputs.last_hidden_state[:, -1, :]
                    batch_output_log_probs = log_softmax(model.lm_head(last_token_hidden_state), dim=-1)
                    batch_output_probs = softmax(model.lm_head(last_token_hidden_state), dim=-1)
                else:
                    outputs = model(**inputs)
                    pred_logits = outputs.logits[:, -1, :].squeeze(1)
                    batch_output_log_probs = log_softmax(pred_logits, dim=-1)
                    batch_output_probs = softmax(pred_logits, dim=-1)
                
            batch_output_log_probs = batch_output_log_probs.to(model.device)
            batch_output_probs = batch_output_probs.to(model.device)

        if log_probs_mode is True:
            batch_cond_2_log_probs = batch_output_log_probs[:, torch.tensor(base_10_tokens).flatten()]
            if batch_cond_2_log_probs.dim() == 1: 
                batch_cond_2_log_probs = batch_cond_2_log_probs.unsqueeze(1)
            log_p_d3_given_d2d1.append(batch_cond_2_log_probs)

        elif log_probs_mode == "both":
            batch_cond_2_log_probs = batch_output_log_probs[:, torch.tensor(base_10_tokens).flatten()]
            batch_cond_2_probs = batch_output_probs[:, torch.tensor(base_10_tokens).flatten()]
            if batch_cond_2_probs.dim() == 1: 
                batch_cond_2_probs = batch_cond_2_probs.unsqueeze(1)
            if batch_cond_2_log_probs.dim() == 1: 
                batch_cond_2_log_probs = batch_cond_2_log_probs.unsqueeze(1)
            log_p_d3_given_d2d1.append(batch_cond_2_log_probs)
            p_d3_given_d2d1.append(batch_cond_2_probs)

        else:
            batch_cond_2_probs = batch_output_probs[:, torch.tensor(base_10_tokens).flatten()]
            p_d3_given_d2d1.append(batch_cond_2_probs)

        if not config.DEBUG_MODE:
            if lm_head_manual:
                vars_to_delete = [inputs, outputs, batch_output_log_probs, batch_output_probs, last_token_hidden_state]
            else:
                vars_to_delete = [inputs, outputs, batch_output_log_probs, batch_output_probs, pred_logits]
            for var in vars_to_delete:
                if var is not None:
                    try:
                        del var
                    except Exception as e:
                        warnings.warn(f"Error deleting variables: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    #calculate joint distributions from conditional distributions
    if log_probs_mode is True:
        log_p_d3_given_d2d1 = [t if t.dim() == 2 else t.unsqueeze(0) for t in log_p_d3_given_d2d1] #aggressive shape normalisation
        log_p_d3_given_d2d1 = torch.cat(log_p_d3_given_d2d1, dim=0)  # (100,10)
        log_p_d1d2d3 = log_p_d3_given_d2d1 + log_p_d1d2.flatten()[:, None]
        log_p_d1d2d3 = log_p_d1d2d3.to("cpu").flatten()
        p_d1d2d3 = torch.exp(log_p_d1d2d3)
        final_answer_probs.extend([t.item() for t in p_d1d2d3])
    elif log_probs_mode == "both":
        # log space
        log_p_d3_given_d2d1 = [t if t.dim() == 2 else t.unsqueeze(0) for t in log_p_d3_given_d2d1] #aggresive shape normalisation
        log_p_d3_given_d2d1 = torch.cat(log_p_d3_given_d2d1, dim=0)  # (100,10)
        log_p_d1d2d3 = log_p_d3_given_d2d1 + log_p_d1d2.flatten()[:, None]
        log_p_d1d2d3 = log_p_d1d2d3.to("cpu").flatten()
        p_d1d2d3_log = torch.exp(log_p_d1d2d3)
        final_answer_probs_log.extend([t.item() for t in p_d1d2d3_log])
        # linear space
        p_d3_given_d2d1 = [t if t.dim() == 2 else t.unsqueeze(0) for t in p_d3_given_d2d1] #aggressive shape normalisation
        p_d3_given_d2d1 = torch.cat(p_d3_given_d2d1, dim=0)  # (100,10)
        p_d1d2d3 = p_d3_given_d2d1 * p_d1d2.flatten().unsqueeze(1)
        p_d1d2d3 = p_d1d2d3.to("cpu").flatten()
        final_answer_probs.extend([t.item() for t in p_d1d2d3])
    else:
        p_d3_given_d2d1 = [t if t.dim() == 2 else t.unsqueeze(0) for t in p_d3_given_d2d1] #aggressive shape normalisation
        p_d3_given_d2d1 = torch.cat(p_d3_given_d2d1, dim=0)  # (100,10)
        p_d1d2d3 = p_d3_given_d2d1 * p_d1d2.flatten().unsqueeze(1)
        p_d1d2d3 = p_d1d2d3.to("cpu").flatten()
        final_answer_probs.extend([t.item() for t in p_d1d2d3])

    if not config.DEBUG_MODE:
        torch.cuda.empty_cache()
        gc.collect()
        clean_up_gpus()


    if log_probs_mode == "both":
        return final_answer_probs, final_answer_probs_log
    else:
        return final_answer_probs