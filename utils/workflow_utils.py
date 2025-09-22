
# Core dependencies
import torch
import gc
import time
import warnings
import os 
import pdb

from vllm import LLM, SamplingParams
from utils.tokenization_setup import *
from utils.dataset_utils import assert_padding_consistency


#lamdas
encode = lambda x, tokenizer: tokenizer.encode(x)
decode = lambda x, tokenizer: tokenizer.decode(x)
decode_no_special = lambda x, tokenizer: tokenizer.decode(x, skip_special_tokens=True)
all_pad = lambda x, pad_token_id: torch.all(x == pad_token_id).item()


#other
def remove_pad_tokens(sequences, tokenizer):
    
    pad_token_id = tokenizer.pad_token_id
    if isinstance(sequences, (list, tuple)):
        return [seq[seq != pad_token_id] for seq in sequences]
    else:
        return sequences[sequences != pad_token_id]

def no_right_or_middle_padding(input_ids, tokenizer):
    """Returns true if sequence has no right or middle padding"""
    if input_ids.dim()!=1: input_ids=input_ids.squeeze(0)
    # Convert to tensor if not already
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    
    # Get positions of all pad tokens
    pad_positions = (input_ids == tokenizer.pad_token_id).nonzero()
    
    if len(pad_positions) == 0:
        return True
        
    # Get positions of all non-pad tokens
    nonpad_positions = (input_ids != tokenizer.pad_token_id).nonzero()
    
    # Check that all pad tokens come before all non-pad tokens
    return all(p < nonpad_positions[0].item() for p in pad_positions)

def no_right_padding(input_ids, tokenizer):
    """Returns true if sequence has no right padding"""
    if input_ids.dim()!=1: input_ids=input_ids.squeeze(0)
    non_pad_indices = (input_ids != tokenizer.pad_token_id).nonzero()
    return non_pad_indices[-1].item() == len(input_ids)-1 #evaluates to true if token at end of sequence is not a pad token

def left_repad_to_length(seq,length,tokenizer):

    non_padded_output_seq = remove_pad_tokens(seq,tokenizer)
    pad_length = length - non_padded_output_seq.shape[0]
    if pad_length<0:
        output_seq = non_padded_output_seq[:length] #truncate
        pad_length=0
    else:
        output_seq = torch.concat([torch.full((pad_length,), tokenizer.pad_token_id,dtype=seq.dtype,device=seq.device),non_padded_output_seq]) #repad

    return output_seq

def batch_pad_inputs(inputs,tokenizer,device):
    max_len = max(seq.shape[0] for seq in inputs)
    padded_inputs = []
    attention_masks = []
    for seq in inputs:
        pad_length = abs(max_len - seq.shape[0])    
        padded_seq = torch.concat([
            torch.full((pad_length,), tokenizer.pad_token_id,dtype=seq.dtype,device=device),
            seq.to(device)
        ])
        attention_mask = torch.where(padded_seq == tokenizer.pad_token_id, 0, 1)
        padded_inputs.append(padded_seq)
        attention_masks.append(attention_mask)

    padded_inputs = torch.stack(padded_inputs)
    attention_masks = torch.stack(attention_masks)

    assert_padding_consistency(padded_inputs, attention_masks, tokenizer, elementwise=True)
    
    
    return padded_inputs, attention_masks

def count_generated_tokens(sequence, end_of_input_idxs, tokenizer):
    """
    Return number of generated tokens in sequence by counting tokens after end of input tokens
    """

    if len(end_of_input_idxs) == 0:
        raise ValueError("end_of_input_idxs is empty")
    else:
        end_idx = end_of_input_idxs[-1]

    # Index after the last end_of_input_idx
    generated = sequence[end_idx+1:]

    # Count non-pad tokens
    if hasattr(generated, "ne"): #check if pytorch tensor
        count = (generated != tokenizer.pad_token_id).sum().item()
    else:
        count = sum([t != tokenizer.pad_token_id for t in generated])

        
    return count




def no_eos_in_sequence(sequence,tokenizer):
    #returns 1 if no eos in sequence after bos, 0 otherwise
    if tokenizer.bos_token_id is not None:
        seq_after_bos = (tokenizer.decode(sequence).split(tokenizer.bos_token))[1] #get sequence after bos token
    else:
        first_non_pad = (sequence != tokenizer.pad_token_id).nonzero()[0][0]
        seq_after_bos = tokenizer.decode(sequence[first_non_pad:])

    return 1 if tokenizer.eos_token not in seq_after_bos else 0

#vllm 
def generate_with_vllm_or_hf(input_ids, attention_mask, gen_params, inference_engine, tokenizer, vllm_model, hf_model, max_new_tokens):

    """
    Do generation with vllm (llm.generate()) or huggingface (model.generate())
    Uses token ids, not text, for I/O. 
    """

    #enforcing again
    tokenizer.padding_side = "left"

    if inference_engine=="hf": assert hf_model is not None, "HF model not found"
    if inference_engine=="vllm": assert vllm_model is not None, "VLLM model not found"

    if inference_engine=="vllm":
        #input ids to text 
        batch_texts=[]
        for i in range(input_ids.shape[0]):
            non_pad_mask = input_ids[i]!=tokenizer.pad_token_id
            non_pad_tokens = input_ids[i][non_pad_mask]
            text = tokenizer.decode(non_pad_tokens,skip_special_tokens=False)
            batch_texts.append(text)

        vllm_bad_words = [tokenizer.decode(a,skip_special_tokens=False) for a in gen_params.get("bad_words_ids",[])]

        #setup sampling params
        sampling_params = SamplingParams(
            temperature=gen_params.get('temperature',0.7),
            top_p=gen_params.get("top_p",0.95),
            top_k=gen_params.get("top_k",20),
            repetition_penalty=gen_params.get("repetition_penalty",1.1),
            bad_words=vllm_bad_words,
            max_tokens=max_new_tokens,
            include_stop_str_in_output=True, # includes string in stop in output
            skip_special_tokens=False, #include special tokens in the output!
        )

        outputs = vllm_model.generate(batch_texts,sampling_params)

        #back to hf 
        sequences = []

        #convert vllm text outputs back to hf token ids
        for i, output in enumerate(outputs):
            full_text = output.outputs[0].text
            
            #tokenize 
            tokens = tokenizer.encode(batch_texts[i]+full_text, return_tensors="pt",add_special_tokens=False)[0]
            sequences.append(tokens)

        #left pad hf token ids 
        max_len = max(seq.shape[0] for seq in sequences)
        padded_sequences = [] 
        for seq in sequences:
            pad_length = max_len - seq.shape[0]
            padded_seq = torch.cat([
                torch.full((pad_length,),tokenizer.pad_token_id,dtype=seq.dtype,device=input_ids.device),
                seq.to(input_ids.device)
            ])
            padded_sequences.append(padded_seq)

        #create similar return object to model.generate() 
        class VLLMGenerateOutput:
            def __init__(self,sequences):
                self.sequences = torch.stack(sequences)
        
        out = VLLMGenerateOutput(padded_sequences)
        return out
    
    #do hf generation 
    else: 
        generate_kwargs = {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "max_new_tokens":max_new_tokens,
            **gen_params, 
        }
        out = hf_model.generate(**generate_kwargs)
        return out


def get_vllm_model_and_tokenizer(
    model_path,
    tokenizer,
    device=None,
    vllm_quantization=None,
    tensor_parallelism=1,
    gpu_memory_utilization=0.6,
    enforce_eager=False,
    max_model_len=32768,
    max_num_seqs=64,
    ):

    vllm_model = LLM(
        model=model_path,
        dtype="auto", 
        tensor_parallel_size=tensor_parallelism,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=vllm_quantization,
        distributed_executor_backend=None,
        enforce_eager=enforce_eager,
        download_dir=os.getenv("HF_CACHE_PATH"),
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        load_format="bitsandbytes",
    )

    vllm_tokenizer = vllm_model.get_tokenizer()
    vllm_tokenizer.pad_token = tokenizer.pad_token
    vllm_tokenizer.padding_side = "left"

    return vllm_model, vllm_tokenizer


def check_model_device(model,device):
    model_device_actual = next(model.parameters()).device
    if model_device_actual!=device:
        raise ValueError(f"Model is on device {model_device_actual} but should be on device {device}")
    else:
        print(f"Model is on correct device {device}")



#utils for force continuation 

def get_end_of_input_idxs(input_ids, tokenizer):
    """
    Gets 'end of input' token indices for each sequence in the batch.
    Returns tensor of indices where the end_of_input pattern ends.
    """

    device = input_ids.device
    
    # Fix the condition logic - use 'in' properly
    if "Qwen2.5" in tokenizer.name_or_path or "simplescaling" in tokenizer.name_or_path:
        end_of_input_ids = qwen_end_of_input_ids
    elif "gemma-2" in tokenizer.name_or_path:
        end_of_input_ids = gemma_end_of_input_ids
    elif "DeepSeek-R1" in tokenizer.name_or_path:
        end_of_input_ids = deepseek_end_of_input_ids
    elif "QwQ" in tokenizer.name_or_path:
        end_of_input_ids = qwq_end_of_input_ids
    elif "Phi" in tokenizer.name_or_path:
        end_of_input_ids = phi_end_of_input_ids
    else:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported for force continuation")
    
    end_of_input_idxs = []
    pattern_len = len(end_of_input_ids)
    
    for seq in input_ids:
        found = False
        # Use sliding window to find pattern
        for i in range(len(seq) - pattern_len + 1):
            if torch.all(seq[i:i+pattern_len] == torch.tensor(end_of_input_ids,device=device)).item():
                # Return the index AFTER the pattern (where generation starts)
                end_of_input_idxs.append(list(range(i,i+pattern_len)))
                found = True
                break
        
        if not found:
            # If pattern not found, assume entire sequence is input
            end_of_input_idxs.append([len(seq)])


            
    return end_of_input_idxs

def remove_last_boxed_tokens(sequence,boxed_token_ids,eoi_idxs,n_tokens_remove=10):

    """
    Truncate sequence at boxed token id and n_tokens before
    """

    assert len(boxed_token_ids)==1, "Workflow supported for single boxed token id only"


    last_boxed_token_idx = torch.where(sequence==boxed_token_ids)[0][-1].item()

    if last_boxed_token_idx < eoi_idxs[0]: #if the boxed token id is found before end of input id
        return sequence[:-n_tokens_remove] #just remove n tokens, not boxed id. hacky. 


    return sequence[:last_boxed_token_idx-n_tokens_remove]



#device management
def assign_devices(num_gpus,inference_engine):
    "Does hf,vllm engine device assignment"

    if num_gpus==0:
        return "cpu",None
    elif num_gpus==1:
        if inference_engine=="hf":
            return "cuda",None #cuda for hf, 
        elif inference_engine=="vllm":
            return "cpu","cuda" #cpu for hf, cuda for vllm
        else:
            return "cuda",None
    else:
        if inference_engine=="hf":
            return "cuda",None #all devices to hf
        elif inference_engine=="vllm":
            if num_gpus==2:
                return "cuda:1","cuda:0" #hf on second gpu,vllm on first gpu
            elif num_gpus==1:
                return None, "cuda", #vllm on gpu, hf on cpu
            elif num_gpus==3:
                return ["cuda:2"],["cuda:0","cuda:1"] #hf on last gpu,vllm on first two
            elif num_gpus==4:
                return ["cuda:2","cuda:3"],["cuda:0","cuda:1"] #hf on first two, vllm on last two
        else:
            return "cuda",None

def get_single_device(devices):
    if torch.cuda.is_available():
        if isinstance(devices, list):
            return devices[0]
        else:
            return devices
    else: 
        return "cpu"


def clean_up_gpus():
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        torch.cuda.set_device(0) #back to 0


def check_for_force_continue_repetition(FORCE_CONTINUE_TOKEN_TRACKER,n_tokens_gen_threshold=50):

    """
    Checks for force continuation with a few different methods
    """
    # token_diff_tracker: list of m tensors, each shape (k,)
    # n_tokens_gen_threshold: int
    # Output: tensor of bools, shape (k,), True if all m diffs < threshold for that sequence


    repetitive_seq_idxs = torch.zeros(len(FORCE_CONTINUE_TOKEN_TRACKER), dtype=torch.bool)
    
    for local_idx, (global_idx, tracker_dict) in enumerate(FORCE_CONTINUE_TOKEN_TRACKER.items()):

        if tracker_dict:  # only check if dict is not empty
            # If any value in the dict is less than or equal to threshold, mark as repetitive
            for v in tracker_dict.values():
                if v <= n_tokens_gen_threshold:
                    repetitive_seq_idxs[local_idx] = True
                    break
    
    return repetitive_seq_idxs

def track_force_continue_generations(sequence,tokenizer,config):

    force_continue_string = config.force_continue
    force_continue_string_tokens = tokenizer.encode(force_continue_string, return_tensors="pt",add_special_tokens=False)[0]

    #type alignment
    if isinstance(sequence, torch.Tensor):
        sequence = tokenizer.decode(sequence,skip_special_tokens=False)
    

    split_sequence = sequence.split(force_continue_string)

    if len(split_sequence) == 1:
        return {} #no force continue string found

    force_continue_tracker = {}
    for split_idx,split in enumerate(split_sequence):
        if split_idx == 0:
            continue #nothing in the first split
        else: 
            n_spilt_tokens = len(tokenizer.encode(split,add_special_tokens=False))
            force_continue_tracker[split_idx] = n_spilt_tokens

    return force_continue_tracker

        
def batch_process_force_end_sequences(model,force_end_inputs,force_end_attention_mask,force_end_gen_params,max_new_tokens,batch_size=None):

    if batch_size is None:
        batch_size = len(force_end_inputs)

    force_end_output_sequences = [] #list of tensors
    force_end_output_scores = [] 

    for i in range(0,len(force_end_inputs),batch_size):
        force_end_inputs_batch = force_end_inputs[i:i+batch_size]
        force_end_attention_mask_batch = force_end_attention_mask[i:i+batch_size]

        
        force_end_outputs_batch = model.generate(input_ids=force_end_inputs_batch,attention_mask=force_end_attention_mask_batch,max_new_tokens=max_new_tokens,**force_end_gen_params)

        force_end_output_sequences.append(force_end_outputs_batch.sequences)
        force_end_output_scores.append(force_end_outputs_batch.scores[0]) #we only need prediction logist for first token

    force_end_output_sequences = torch.vstack(force_end_output_sequences)
    force_end_output_scores = torch.vstack(force_end_output_scores)    

    return force_end_output_sequences, force_end_output_scores #return tensor
    
