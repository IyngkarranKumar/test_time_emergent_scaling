import os, sys, pdb

import logging, pdb, sys,glob, subprocess
from datetime import datetime
import gc, importlib, warnings, copy
import pickle, time
import torch, numpy as np, pandas as pd
import utils, conf, tempfile
if torch.cuda.is_available(): import pynvml

from datasets import load_dataset
from torch.nn.functional import softmax 
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, AwqConfig, CompressedTensorsConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams #load vllm at very last stage

importlib.reload(utils)
importlib.reload(conf); from conf import config

def main():
    import os 
    from dotenv import load_dotenv

    #shell env setup
    if os.environ.get("node_type")=="local":
        dot_env_path="envs/.env.local"
    elif os.environ.get("node_type")=="mlp_head":
        dot_env_path="envs/.env.mlp_head"
    elif os.environ.get("node_type")=="mlp_compute":
        dot_env_path="envs/.env.mlp_compute"
    elif os.environ.get("node_type")=="vastai":
        dot_env_path="envs/.env.vastai"
    elif os.environ.get("node_type")=="eddie_head":
        dot_env_path="envs/.env.eddie_head"
    elif os.environ.get("node_type")=="eddie_compute":
        dot_env_path="envs/.env.eddie_compute"
    else:
        raise ValueError(f"Unknown node type: {os.environ.get('node_type')}")
    load_dotenv(dot_env_path)

    

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_dir="logs"
    os.makedirs(log_dir,exist_ok=True)
    logfile=f"{log_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(logfile)], 
    )
    logger = logging.getLogger(__name__)


    logger.info(f"Working on {os.getenv('node_type')}")
    logger.info(f"HF CACHE, HF HOME, HF DATASETS CACHE: {os.getenv('HF_CACHE_PATH')}")
    logger.info(f"HF_DATASETS_OFFLINE: {os.getenv('HF_DATASETS_OFFLINE')}")
    logger.info(f"TRANSFORMERS_OFFLINE: {os.getenv('TRANSFORMERS_OFFLINE')}")

    #set all seeds for deterministic generation
    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Clear CUDA memory
    if torch.cuda.is_available():
        logger.info("Clearing CUDA memory")

        # shell script to clear gpu memory first
        logger.info("Clearing CUDA memory with shell script")
        script_path = os.path.join(os.path.dirname(__file__), "scripts/clear_gpu.sh")
    
        if os.path.exists(script_path):
            try:
                result = subprocess.run(["bash", script_path], 
                                    capture_output=True, 
                                    text=True, 
                                    timeout=30)
                logger.info(result.stdout)
                if result.stderr:
                    logger.info("Warnings:", result.stderr)
            except subprocess.TimeoutExpired:
                logger.info("GPU clearing script timed out")
            except Exception as e:
                logger.info(f"Error running GPU clear script: {e}")



        # Clear CUDA cache
        logger.info("Clearing CUDA cache")
        utils.clean_up_gpus()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Clear CUDA memory
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass
                
    
    utils.clean_up_gpus() # Final cache clear

    all_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    logger.info(f"CUDA_VISIBLE_DEVICES: {all_cuda_devices}")



    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_vram_info = {i:None for i in range(num_gpus)}
        logger.info(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_vram_info[i] = mem_info.total // 1024**3
            logger.info(f"GPU {i} Memory - Total: {mem_info.total // 1024**3}GB, Used: {mem_info.used // 1024**3}GB, Free: {mem_info.free // 1024**3}GB")

        #include old profiling function
        logger.info(f"GPU utilisation: {utils.gpu_utilisation(all=True)}")

        # Cleanup
        pynvml.nvmlShutdown()


    #those god damn warnings 
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

    logger.info("CONFIG DETAILS:")
    logger.info(f"\n{config}\n") #using repr(?) method
    if torch.cuda.is_available():
        logger.info(f"Number of GPUs: {num_gpus}")



    #%% tokenization and model setup




    if 1: #tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,cache_dir=os.getenv("HF_CACHE_PATH"),local_files_only=True if int(os.getenv("TRANSFORMERS_OFFLINE",0))==1 else False)

        #token setup 
        force_end=config.force_end
        force_continue=config.force_continue

        if tokenizer.bos_token is None or tokenizer.bos_token==tokenizer.eos_token: #handling bos token for models that don't have it
            logger.info(f"No unique bos token found for model {config.tokenizer_name} - bos token is: {tokenizer.bos_token}")

        if tokenizer.pad_token is None or tokenizer.pad_token==tokenizer.eos_token: #handling pad token for models that don't have it, or set to same as eos
            logger.info(f"No unique pad token found for tokenizer {config.tokenizer_name}, adding pad token pad token distinct from eos token")

            if "DeepSeek-R1" in config.tokenizer_name: #claude suggested 
                tokenizer.pad_token,tokenizer.pad_token_id = utils.deepseek_pad_token,utils.deepseek_pad_token_id
            elif "QwQ" in config.tokenizer_name:
                tokenizer.pad_token,tokenizer.pad_token_id = utils.qwq_pad_token,utils.qwq_pad_token_id
            elif "Qwen2.5" in config.tokenizer_name:
                tokenizer.pad_token,tokenizer.pad_token_id = utils.qwen25_pad_token,utils.qwen25_pad_token_id
            else: 
                raise ValueError(f"No unique pad token found for tokenizer {config.tokenizer_name}, and tokenizer is not the deepseek, qwq or qwen2.5 tokenizer")

        tokenizer.padding_side = "left"

        logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
        logger.info(f"Pad token: {tokenizer.pad_token}")
        logger.info(f"EOS token: {tokenizer.eos_token}")
        logger.info(f"BOS token: {tokenizer.bos_token}")
        

    

    if 1:     #model precision setup (quantization)

        if os.getenv("node_type") == "local" or not torch.cuda.is_available():
            quantization_config = None
            vllm_quantization = None
            torch_dtype=None
        else:
            # HuggingFace/Transformers quantization
            if config.quantization == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16
                    bnb_4bit_use_double_quant=True,       # Double quantization for better accuracy
                    bnb_4bit_quant_type="nf4"             # Normal Float 4-bit
                )
                torch_dtype=None
            elif config.quantization == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                torch_dtype=None
            else:
                quantization_config = None
                torch_dtype=torch.float16
            
            # vLLM quantization 
            if config.inference_engine == "vllm":
                if config.quantization == 4:
                    vllm_quantization = "awq"  # not sure if this works without awq model. 
                elif config.quantization == 8:
                    vllm_quantization = "bitsandbytes"   # or None for auto int8
                else:
                    vllm_quantization = None
            else:
                vllm_quantization = None



    logger.info(f"Quantization config: {quantization_config}")
    logger.info(f"VLLM quantization: {vllm_quantization}")
        

    if 1: #device mapping
        #model map (across devices) setup
        num_gpus = torch.cuda.device_count()

        if num_gpus==1:
            vllm_devices=["cuda:0"]
            hf_devices=["cuda:0"]
        else: 
            vllm_devices = [f"cuda:{i}" for i in range(num_gpus//2)] #first half to vllm
            hf_devices = [f"cuda:{i}" for i in range(num_gpus//2,num_gpus)] #second half to hf

        #both of these work I think
        if torch.cuda.is_available():
            gpu_vram_with_buffer = gpu_vram_info[0]-1 #get Vram, incl. buffer
        hf_device_map="auto"
        hf_device_max_memory = {**{int(device.split(":")[-1]): "0GB" for device in vllm_devices}, **{int(device.split(":")[-1]): str(gpu_vram_with_buffer)+"GB" for device in hf_devices}}
        hf_device_max_memory["cpu"]="50GB"


        tensor_parallelism = len(vllm_devices) #vllm will distribution across first n gpus, with n set here

            
        logger.info(f"VLLM devices: {vllm_devices}")
        logger.info(f"HF devices: {hf_devices}")
        logger.info(f"HF device map: {hf_device_map}")
        logger.info(f"HF device max memory: {hf_device_max_memory}")
            
    
        if torch.cuda.is_available():
            default_device = hf_devices[0]
            default_force_end_device = hf_devices[0]
            if config.inference_engine=="vllm":
                default_force_continue_device = vllm_devices[0]
                default_normal_gen_device = vllm_devices[0]
            else:
                default_force_continue_device = hf_devices[0]
                default_normal_gen_device = hf_devices[0]
        else: 
            default_device = "cpu"
            default_force_end_device = "cpu"
            default_force_continue_device = "cpu"
            default_normal_gen_device = "cpu"


    if 1: #model loading
        if config.inference_engine=="vllm":

            #get path for vllm model

            #offline 
            if int(os.getenv("TRANSFORMERS_OFFLINE",0))==1:
                cache_name = config.model_name.replace('/', '--')
                if not cache_name.startswith('models--'):
                    cache_name = 'models--' + cache_name

                # Find the snapshot directory if it exists
                snapshot_dir = f"{os.getenv('HF_CACHE_PATH')}/{cache_name}/snapshots"

                snapshot_hash = os.listdir(snapshot_dir)[0]
                model_path = f"{snapshot_dir}/{snapshot_hash}"
            else:
                model_path = config.model_name #hopefully works as normal
                

            logger.info(f"Initialising VLLM engine from {model_path}")

            #setting eager execution or not (false is higher performance)
            if os.getenv("node_type") == "vastai":
                if config.DEBUG_MODE or len(vllm_devices) > 1:
                    enforce_eager = True
                else:
                    enforce_eager = False
            elif os.getenv("node_type") == "mlp_compute":
                if config.DEBUG_MODE or len(vllm_devices) > 1:
                    enforce_eager = True
                else:
                    enforce_eager = False
            else:
                enforce_eager = True

            logger.info(f"VLLM eager execution: {enforce_eager}")
            with utils.gpu_util_manager("Loading VLLM model",logger):
                vllm_model, vllm_tokenizer = utils.get_vllm_model_and_tokenizer(
                    model_path=model_path,
                    device=vllm_devices[0],
                    tokenizer=tokenizer,
                    vllm_quantization=vllm_quantization,
                    tensor_parallelism=tensor_parallelism,
                    gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                    enforce_eager=enforce_eager,
                    max_model_len=config.max_model_len,
                    max_num_seqs=config.vllm_max_num_seqs,
                )
            logger.info(f"VLLM model loaded")



        else:
            vllm_model = None


        utils.clean_up_gpus()
        
        if 1: # HF MODEL LOADING


            with utils.gpu_util_manager("Loading HF model",logger):
                    model = AutoModelForCausalLM.from_pretrained(
                                config.model_name,
                                quantization_config=quantization_config,
                                device_map=hf_device_map,
                                max_memory=hf_device_max_memory,
                                cache_dir=os.getenv("HF_CACHE_PATH"),
                                low_cpu_mem_usage=False,
                                local_files_only=True if int(os.getenv("TRANSFORMERS_OFFLINE",0))==1 else False,
                                torch_dtype=torch_dtype,
                            )
    
            
            model.config.pad_token_id = tokenizer.pad_token_id #make sure this checks
            num_layers, hidden_size, vocab_size, num_attention_heads = utils.get_model_info(model)
            logger.info(f"HF model loaded on device: {model.hf_device_map}")
            logger.info(f"Model info: num_layers: {num_layers}, hidden_size: {hidden_size}, vocab_size: {vocab_size}, num_attention_heads: {num_attention_heads}")

            if config.model_name != config.tokenizer_name: #just for debug frankenstein models
                logger.info(f"Model name: {config.model_name}, Tokenizer name: {config.tokenizer_name}")
                logger.info(f"Model name is not equal to tokenizer name: {config.model_name} != {config.tokenizer_name}")
                assert os.getenv("node_type") not in ["vastai", "mlp_compute"], "Must be a local debug run"

                # Resize model embedding layer if needed
                if hasattr(tokenizer, "vocab_size") and hasattr(model, "resize_token_embeddings"):
                    old_num_tokens = model.get_input_embeddings().num_embeddings
                    new_num_tokens = len(tokenizer)
                    if new_num_tokens != old_num_tokens:
                        logger.info(f"Resizing model embeddings from {old_num_tokens} to {new_num_tokens}")
                        model.resize_token_embeddings(new_num_tokens)
                        
                



        utils.clean_up_gpus()


    try:
        model_generation_config=GenerationConfig.from_pretrained(config.model_name,cache_dir=os.getenv("HF_CACHE_PATH"))
        logger.info(f"Model generation config: {model_generation_config}")
    except:
        model_generation_config=None
        logger.info(f"No model generation config found for model {config.model_name}")

    gen_params={
        "temperature":model_generation_config.temperature if model_generation_config else config.default_temperature,
        "top_p":model_generation_config.top_p if model_generation_config else config.default_top_p,
        "top_k": model_generation_config.top_k if model_generation_config else config.default_top_k,
        "repetition_penalty":model_generation_config.repetition_penalty if model_generation_config else config.default_repetition_penalty,
        "do_sample":config.do_sample,
        "return_dict_in_generate":config.return_dict_in_generate,
        "eos_token_id":tokenizer.eos_token_id,
        "pad_token_id":tokenizer.pad_token_id,
        "use_cache":True, #use kv cache
        "output_scores":False, #only true for last step
        "output_hidden_states":False,
        "output_attentions":False,
    }

    force_end_gen_params = copy.deepcopy(gen_params)
    force_end_gen_params["output_scores"] = True

    force_continue_gen_params = copy.deepcopy(gen_params)
    #force_continue_gen_params["frequency_penalty"] = 1.5
    #force_continue_gen_params["repetition_penalty"] = 1.5


    


    #%%dataset 

    logger.info(f"Loading dataset {config.dataset_name}")

    #load dataset
    ds=utils.DatasetSetup(config.dataset_name,num_samples=config.num_samples,config=config)
    dataset_samples = ds.load_standardized_dataset(config)
    num_samples = len(dataset_samples)
    logger.info(f"Number of samples: {num_samples}")


    

    #%% main


    print("\n" + "="*50)
    print("✨ STARTING GENERATION ✨")
    print("="*50 + "\n")

    last_save_time = time.time()


    #setup save data structures 
    SAVE_DATA={} #init OUTPUT DATA STRUCTURE


    if config.SAVE_BOOL:
        utils.save_data(config, config.save_dir, config_save=True)

    #pre-computes and put on correct device
    force_end_ids = tokenizer.encode(force_end, return_tensors='pt',add_special_tokens=False).to(default_force_end_device).squeeze(0) #hf_device, 1 dim
    force_continue_ids = tokenizer.encode(force_continue, return_tensors='pt',add_special_tokens=False).to(default_force_continue_device).squeeze(0) #1 dim
    boxed_token_ids = torch.tensor(tokenizer.encode(config.answer_indicator_token,add_special_tokens=False),device=default_force_continue_device)

    #set token budgets 

    token_budgets = config.token_budgets #not sure why we do this, but keep for now
    for token_iter, token_budget in enumerate(token_budgets):

        logger.info(f'Generating for token budget {token_budget}')
        token_budget_start_time = time.time()
        
        if 1: #token budget inits
            SAVE_DATA[token_budget]={}


            if config.USE_TOKEN_BUDGET_BATCH_SIZE_MAPPING:
                if token_budget in config.token_batch_mapping:
                    batch_size, solution_set_batch_size = config.token_batch_mapping[token_budget]
                else:
                    batch_size, solution_set_batch_size = config.token_batch_mapping["default"]
            else:
                batch_size = config.batch_size
                solution_set_batch_size = config.solution_set_batch_size

            # Create batches from samples
            sample_batches = utils.create_batches(list(dataset_samples), batch_size)

            #save init 
            batches_in_save_data = []


        for batch_idx, sample_batch in enumerate(tqdm(sample_batches, desc=f"Processing batches (budget={token_budget})", leave=False)):

            if config.SAVE_BOOL:
                batches_in_save_data.append(batch_idx)

            logger.info(f"Doing generation for batch {batch_idx + 1}/{len(sample_batches)}")
            
            if 1: #genreation for batch

                if 1: #batch inits
                    logger.info(f'Processing batch {batch_idx + 1}/{len(sample_batches)}')
                    
                    # Pre-compute all questions and their tokenized versions
                    batch_questions = [sample['question'] for sample in sample_batch]
                    batch_input_tokens, question_lengths_tokens = utils.prepare_batch_inputs(batch_questions, tokenizer, config, device=default_device,with_chat_template=True,force_end_prompt=config.force_end)
                    input_prompts_max_len = question_lengths_tokens.max().item()


                    logger.info(f"Prompt lengths (tokens): {question_lengths_tokens}")
                    
                    # Expand batch size by num_completions
                    expanded_batch = []
                    for sample in sample_batch:
                        for _ in range(config.num_completions):
                            expanded_batch.append(sample.copy())
                    
                    batch_size_actual = len(expanded_batch)
                    
                    # Initialize output data for this batch
                    batch_start_idx = batch_idx * batch_size
                    for i, sample in enumerate(expanded_batch):
                        sample_idx = batch_start_idx + (i // config.num_completions)
                        completion_idx = i % config.num_completions
                        if sample_idx not in SAVE_DATA[token_budget]:
                            SAVE_DATA[token_budget][sample_idx] = {}
                        if 'completions' not in SAVE_DATA[token_budget][sample_idx]:
                            SAVE_DATA[token_budget][sample_idx]['completions'] = {}
                        SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx] = {'text': []}

                    # Expand input tokens for multiple completions
                    expanded_input_ids = []
                    expanded_attention_mask = []
                    for i in range(len(batch_input_tokens['input_ids'])):
                        for _ in range(config.num_completions):
                            expanded_input_ids.append(batch_input_tokens['input_ids'][i])
                            expanded_attention_mask.append(batch_input_tokens['attention_mask'][i])

                    
                    # Track completion status for each sample
                    batch_exit_conditions = torch.zeros(batch_size_actual, dtype=torch.bool, device=default_device)

                    # Track generation tokens for each sample in batch
                    n_generated_tokens = torch.zeros(batch_size_actual, device=default_device)
                    n_generated_tokens_prev = torch.zeros(batch_size_actual, device=default_device)

                    #initialise tracker for checking for force continue repetition
                    GENERATED_TOKEN_DIFF_TRACKER = []
                    repetitive_seq_idxs = [] #initialising as empty list


            if 1: #generation loop 
                while_loop_idx = 0

                while not batch_exit_conditions.all():

                    if 1: #loop inits 

                        #initialise batch input tokens for first generation
                        if (n_generated_tokens==0).all().item()==True:
                            #just set input ids - we set the attention masks in individual generation loops
                            batch_input_tokens = torch.stack(expanded_input_ids)
                        #else batch_input_tokens set by at end of the last loop
                            
                        #init data structures for output of this generation loop
                        max_new_generated_tokens = int(max(
                            force_continue_ids.shape[0]+int(token_budget*config.max_new_tokens_frac), #force continue
                            force_end_ids.shape[0]+config.final_answer_budget, #
                            int(token_budget*config.max_new_tokens_frac))) #normal gen

                        current_max_seq_len = (batch_input_tokens != tokenizer.pad_token_id).sum(dim=1).max().item()

                        max_possible_len = current_max_seq_len + max_new_generated_tokens + config.safety_buffer

                        #init new sequences
                        new_sequences = torch.full((batch_input_tokens.shape[0], max_possible_len), tokenizer.pad_token_id, dtype=batch_input_tokens.dtype, device=default_device,requires_grad=False)


                        #get end of input tokens for each sequence 
                        end_of_input_idxs = utils.get_end_of_input_idxs(batch_input_tokens,tokenizer) #list of lists with end of input tokens

                    #padding sanity check 
                    pad_counts = (batch_input_tokens == tokenizer.pad_token_id).sum(dim=1)
                    batch_token_length = batch_input_tokens.shape[1]

                    if 1: #Allocate samples (and consistency check) between inactive, force end, force continue, and normal gen

                        active_indices = torch.where(~batch_exit_conditions)[0]
                        active_input_tokens = batch_input_tokens[active_indices]
                        active_n_tokens = n_generated_tokens[active_indices]
                        active_end_of_input_idxs = [end_of_input_idxs[i] for i in active_indices]
                        
                        
                        # Determine sample types

                        #check if eos (not including in the user prompt)
                        has_eos = torch.zeros(len(active_input_tokens), dtype=torch.bool, device=default_device)
                        for i in range(len(active_input_tokens)):
                            seq_eoi_idxs = active_end_of_input_idxs[i] #list
                            first_generated_token_idx = seq_eoi_idxs[-1]+1
                            has_eos[i] = torch.any(active_input_tokens[i, first_generated_token_idx:] == tokenizer.eos_token_id) #check if seuqnece has eos in generated sequence

                        exceeds_budget = active_n_tokens >= token_budget
                        #exceeds_budget = active_n_tokens + int(token_budget*config.max_new_tokens_frac) > token_budget
                        
                        # Prioritize force_end, then force_continue, then normal generation

                        
                        force_end_mask = []
                        force_continue_mask = []
                        normal_gen_mask = []
                        for i in range(len(active_input_tokens)):
                            if exceeds_budget[i]:
                                force_end_mask.append(i)
                            elif len(repetitive_seq_idxs) > 0 and repetitive_seq_idxs[i]:
                                force_end_mask.append(i)
                            elif has_eos[i]:
                                force_continue_mask.append(i)
                            else:
                                normal_gen_mask.append(i)
                        force_end_mask = torch.tensor(force_end_mask, dtype=torch.long, device=default_device)
                        force_continue_mask = torch.tensor(force_continue_mask, dtype=torch.long, device=default_device)
                        normal_gen_mask = torch.tensor(normal_gen_mask, dtype=torch.long, device=default_device)
                        
                        logger.info(f"Inactive samples: {torch.where(batch_exit_conditions)[0].tolist()}, Force continue: {active_indices[force_continue_mask].tolist()}, Force end: {active_indices[force_end_mask].tolist()}, Normal gen: {active_indices[normal_gen_mask].tolist()}")
                        
                        # Update exit conditions
                        batch_exit_conditions[active_indices[force_end_mask]] = True

                        # Assert all samples are accounted for
                        total_samples = batch_input_tokens.shape[0]
                        inactive_samples = torch.where(batch_exit_conditions)[0]
                        active_force_continue = active_indices[force_continue_mask]
                        active_force_end = active_indices[force_end_mask] 
                        active_normal_gen = active_indices[normal_gen_mask]
                        
                        all_samples = torch.cat([
                            inactive_samples,
                            active_force_continue,
                            active_force_end,
                            active_normal_gen
                        ])
                        all_samples_sorted = torch.unique(torch.sort(all_samples)[0])
                        expected_samples = torch.arange(total_samples, device=utils.get_single_device(hf_devices))
                        assert torch.equal(all_samples_sorted, expected_samples), "Not all samples accounted for in generation masks"
               
                    if 1: #FORCE END GENERATION BLOCK

                        logger.info(f"GPU utilisation before force end: {utils.gpu_utilisation()}")


                        # Process force end samples
                        if len(force_end_mask) > 0:
                            force_end_inputs = []
                            
                            #add force end seq
                            for local_idx in force_end_mask:
                                input_seq = active_input_tokens[local_idx]
                                input_seq = input_seq.to(default_force_end_device) #move to hf device
                                input_seq = torch.concat([
                                    input_seq.unsqueeze(0),
                                    force_end_ids.unsqueeze(0)
                                ], dim=1)
                                force_end_inputs.append(input_seq.squeeze(0))
                            
                            # Batch pad force end inputs
                            force_end_batch_inputs, force_end_attention = utils.batch_pad_inputs(force_end_inputs,tokenizer,default_force_end_device)

                            utils.clean_up_gpus() 


                            with utils.gpu_util_manager("Force end generation step", logger):
                                with utils.timer("Force end generation step", logger):
                                    force_end_output_sequences,force_end_output_scores = utils.auto_force_end_batch_scale_oom_guard(
                                        initial_size = len(force_end_batch_inputs),
                                        operation_func = utils.batch_process_force_end_sequences,
                                        logger = logger, 
                                        description = "Force end generation step",
                                        scale_factor = 0.6,
                                        min_size = 1,
                                        model = model,
                                        force_end_inputs = force_end_batch_inputs,
                                        force_end_attention_mask = force_end_attention,
                                        force_end_gen_params = force_end_gen_params,
                                        max_new_tokens = config.final_answer_budget,
                                    )
                                        
                                        
                            force_end_output_sequences = force_end_output_sequences.to(new_sequences.device) #move to correct device
                            
                            # Update outputs for force end samples
                            for batch_local_idx, local_idx in enumerate(force_end_mask):
                                global_idx = active_indices[local_idx]
                                sample_idx = batch_start_idx + (global_idx // config.num_completions)
                                completion_idx = global_idx % config.num_completions
                                
                                raw_output_seq = force_end_output_sequences[batch_local_idx]
                                output_seq = utils.left_repad_to_length(raw_output_seq,new_sequences.shape[1],tokenizer)
                                output_seq = output_seq.to(new_sequences.device,dtype=new_sequences.dtype)
                                
                                new_sequences[global_idx] = output_seq #add to new sequneces, respsecting left padding
                                
                                #add text to save object
                                output_text = tokenizer.decode(output_seq)
                                SAVE_DATA[token_budget][sample_idx.item()]['completions'][completion_idx.item()]['text'].append(output_text)
                                
                                #add prediction logits to save object
                                SAVE_DATA[token_budget][sample_idx.item()]['completions'][completion_idx.item()]['logits'] = force_end_output_scores[batch_local_idx].cpu()

                            #clean up
                            if not config.DEBUG_MODE:
                                del force_end_batch_inputs, force_end_attention, force_end_output_sequences, force_end_output_scores
                                gc.collect()
                                utils.clean_up_gpus()

                    if 1: #FORCE CONTINUE GENERATION BLOCK
                        
                        # Process force continue samples
                        if len(force_continue_mask) > 0:
                            force_continue_inputs = []

                            #append force continue tokens
                            for local_idx in force_continue_mask:
                                input_seq = active_input_tokens[local_idx]
                                seq_eoi_idxs = active_end_of_input_idxs[local_idx] #list
                                input_seq = input_seq.to(default_force_continue_device) #move to deviec that vllm on, if vllm

                                # Remove all tokens after first eos
                                eos_indices_all = torch.where(input_seq == tokenizer.eos_token_id)[0]
                                eos_indices_generated = eos_indices_all[eos_indices_all > seq_eoi_idxs[-1]] #eos tokens present in the sequence, after input

                                #append 
                                if len(eos_indices_generated) > 0: #if there are eos tokens after the input prompt
                                    first_eos_index = eos_indices_generated[0]


                                    if config.REMOVE_ANSWER_AT_CONTINUATION:

                                        input_seq = utils.remove_last_boxed_tokens(input_seq,boxed_token_ids=boxed_token_ids,eoi_idxs=seq_eoi_idxs,n_tokens_remove=config.n_tokens_remove_answer)

                                    else:
                                        input_seq = input_seq[:first_eos_index] #remove all tokens including and after first eos in generated sequence #method of removing up to boxed


                                force_continue_seq = torch.concat([
                                    input_seq.unsqueeze(0),
                                    force_continue_ids.unsqueeze(0)
                                ], dim=1) #shape = (1,seq_len)
                                
                                if not utils.no_right_or_middle_padding(force_continue_seq, tokenizer):
                                    warnings.warn(f"Right or middle padding detected in INPUT to force continue generation sample {local_idx}. Attempting to remove...")
                                    force_continue_seq = utils.remove_pad_tokens(force_continue_seq, tokenizer)
                                
                                force_continue_inputs.append(force_continue_seq.squeeze(0) if force_continue_seq.dim()!=1 else force_continue_seq)
                            
                            # left batch pad force continue inputs
                            force_continue_batch_inputs, force_continue_attention = utils.batch_pad_inputs(force_continue_inputs,tokenizer,default_force_continue_device)

                            #generation step
                            with utils.gpu_util_manager("Force continue generation step", logger):
                                with utils.timer("Force continue generation step", logger): #force continue
                                    force_continue_outputs = utils.generate_with_vllm_or_hf(
                                        input_ids = force_continue_batch_inputs, 
                                        attention_mask = force_continue_attention,
                                        gen_params = force_continue_gen_params, 
                                        inference_engine = config.inference_engine, 
                                        tokenizer = tokenizer, 
                                        vllm_model = vllm_model, #initialised earlier
                                        hf_model = model,
                                        max_new_tokens = int(token_budget*config.max_new_tokens_frac)
                                    )
                                    

                            # Update outputs for force continue samples
                            force_continue_output_sequences = force_continue_outputs.sequences.to(new_sequences.device)
                            for batch_local_idx, local_idx in enumerate(force_continue_mask):
                                global_idx = active_indices[local_idx]
                                raw_output_seq = force_continue_output_sequences[batch_local_idx]

                                output_seq = utils.left_repad_to_length(raw_output_seq,new_sequences.shape[1],tokenizer)
                                output_seq = output_seq.to(new_sequences.device,dtype=new_sequences.dtype)
                                
                                new_sequences[global_idx] = output_seq #add to new sequences, respecting left padding
                                
                                


                            #clean up force continue batch, attention, and outputs
                            if not config.DEBUG_MODE:
                                del force_continue_batch_inputs, force_continue_attention, force_continue_outputs
                                gc.collect()
                                utils.clean_up_gpus()

                    if 1:  #NORMAL GENERATION BLOCK
                            if len(normal_gen_mask) > 0:
                                normal_inputs = []
                                for local_idx in normal_gen_mask:
                                    input_seq = active_input_tokens[local_idx]
                                    if not utils.no_right_or_middle_padding(input_seq, tokenizer):
                                        warnings.warn(f"Right or middle padding detected in INPUT to normal generation sample {local_idx}")
                                        input_seq = utils.remove_pad_tokens(input_seq, tokenizer) #remove padding tokens
                                    normal_inputs.append(input_seq) 
                                
                                # Batch pad normal inputs
                                normal_batch_inputs, attention_mask = utils.batch_pad_inputs(normal_inputs,tokenizer,default_normal_gen_device)
                                
                                with utils.gpu_util_manager("Normal generation step", logger):
                                    with utils.timer("Normal generation step", logger): #normal gen
                                        normal_outputs = utils.generate_with_vllm_or_hf(
                                            input_ids = normal_batch_inputs, 
                                            attention_mask = attention_mask,
                                            gen_params = gen_params, 
                                            inference_engine = config.inference_engine, 
                                            tokenizer = tokenizer, 
                                            vllm_model = vllm_model, 
                                            hf_model = model, 
                                            max_new_tokens = int(token_budget*config.max_new_tokens_frac),
                                        )

                                # set output seq to lenght of new_sequences
                                normal_output_sequences = normal_outputs.sequences.to(new_sequences.device)
                                for batch_local_idx, local_idx in enumerate(normal_gen_mask):
                                    global_idx = active_indices[local_idx]
                                    raw_output_seq = normal_output_sequences[batch_local_idx]
                    

                                    output_seq = utils.left_repad_to_length(raw_output_seq,new_sequences.shape[1],tokenizer)

                                    output_seq = output_seq.to(new_sequences.device,dtype=new_sequences.dtype)
                                    new_sequences[global_idx.item()] = output_seq
                                

                                #clean up normal inputs, attention, and outputs
                                if not config.DEBUG_MODE:
                                    del normal_batch_inputs, attention_mask, normal_outputs
                                    gc.collect()
                                    torch.cuda.empty_cache()

                    if 1: #post generation loop idx processing

                        FORCE_CONTINUE_N_TOKEN_TRACKER = {}

                        #save previous n_generated_tokens
                        n_generated_tokens_prev = n_generated_tokens.detach().clone()

                        new_sequences_eoi_idxs = utils.get_end_of_input_idxs(new_sequences,tokenizer)

                        # Update token counts for all active samples
                        for local_idx, global_idx in enumerate(active_indices):
                            seq = new_sequences[global_idx]
                            n_generated_tokens[global_idx] = utils.count_generated_tokens(seq,new_sequences_eoi_idxs[global_idx],tokenizer) #update n generated tokens 

                            if config.CHECK_FOR_FORCE_CONTINUE_REPETITIONS:
                                force_continue_token_tracker = utils.track_force_continue_generations(seq,tokenizer,config)
                                FORCE_CONTINUE_N_TOKEN_TRACKER[global_idx.item()] = force_continue_token_tracker
                                

                        logger.info(f"n_generated_tokens after generation step: {[a.item() for a in n_generated_tokens]} \n")
                        token_diff = n_generated_tokens-n_generated_tokens_prev

                        if config.CHECK_FOR_FORCE_CONTINUE_REPETITIONS:
                            repetitive_seq_idxs = utils.check_for_force_continue_repetition(FORCE_CONTINUE_N_TOKEN_TRACKER,n_tokens_gen_threshold=config.force_continue_repetition_threshold)
                        
                        if torch.any(repetitive_seq_idxs): #debug
                            for local_idx, global_idx in enumerate(active_indices):
                                if repetitive_seq_idxs[local_idx]:
                                    logger.info(f"Sequence {global_idx} identified as repeating. Moving to force end")


                        if torch.all(token_diff==0):
                            logger.info("No change in n_generated_tokens, breaking")
                            sys.exit("No change in n_generated_tokens. Killing process")


                        if config.APPEND_EOS_TOKEN:
                            for local_idx in normal_gen_mask:
                                global_idx = active_indices[local_idx]
                                seq = new_sequences[global_idx]
                                seq[-1] = tokenizer.eos_token_id
                                new_sequences[global_idx] = seq

                        if config.DEBUG_MODE: #for debugging purposes only
                            batch_input_tokens_prev = batch_input_tokens.detach().clone()
                            new_sequences_prev = new_sequences.detach().clone()

                        batch_input_tokens = new_sequences.detach().clone()



                        while_loop_idx += 1        

                    if 1: #clean up

                        if not config.DEBUG_MODE:
                            gc.collect()
                            utils.clean_up_gpus()

                        logger.info(f"GPU utilisation after batch: {utils.gpu_utilisation(all=True)}")


            logger.info(f"Finished generation for batch {batch_idx + 1}/{len(sample_batches)}")

            logger.info(f"Doing scoring for batch {batch_idx + 1}/{len(sample_batches)}")

            if 1: #scoring for batch

                if batch_idx == 0:
                    SAVE_DATA[token_budget]['metrics'] = {
                        'answer_score': [],
                        'answer_probability': [],
                        'answer_entropy': [],
                        'answer_ranking': [],
                        'solution_set_distribution': [],
                    } #init data structure for this token budget

                if 1: #scoring for batch
                    batch_scoring_start_time = time.time()

                    batch_metric_data = {
                        'answer_score': [],
                        'answer_probability': [],
                        'answer_entropy': [],
                        'answer_ranking': [],
                        'solution_set_distribution': [],
                    }

                    if config.normalise_over_solution_set:

                        solution_set = [str(a) for a in ds.solution_set]
                        solution_set_tokens = []
                        for answer in solution_set:
                            answer_tokens = tokenizer(str(answer).strip(), return_tensors='pt',add_special_tokens=False)['input_ids'][0]
                            solution_set_tokens.append(answer_tokens)

                            
                        batch_sample_idxs = list(range(batch_idx*batch_size, (batch_idx+1)*batch_size))
                        for sample_idx in tqdm(batch_sample_idxs, desc=f"Processing batch samples (budget={token_budget})"):

                            if config.dataset_name=="Idavidrein/gpqa":
                                ground_truth_answer = dataset_samples[sample_idx]['MCQ answer']
                            else:
                                ground_truth_answer = dataset_samples[sample_idx]['answer']

                            ground_truth_tokens = tokenizer(ground_truth_answer.strip(), return_tensors='pt',add_special_tokens=False)['input_ids'][0]

                            _ground_truth_idx = np.where(solution_set==np.array(ground_truth_answer))

                            try: 
                                ground_truth_idx = _ground_truth_idx[0].item()
                            except:
                                raise ValueError(f"Ground truth answer {ground_truth_answer} not in solution set")

                            for completion_idx in tqdm(range(config.num_completions), desc="Processing completions)",leave=False):

                                output_text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text']
                                final_text_output = output_text[-1]
                                reasoning_trace, model_output_raw = ''.join(final_text_output.rsplit(force_end)[:-1]), final_text_output.rsplit(force_end)[-1]
                                model_answer = model_output_raw.split("}")[0].strip()
                                first_token_logits = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['logits'].to('cpu') #we only save first token logits

                                if config.dataset_name=="Maxwell-Jia/AIME_2024" or config.dataset_name=="math-ai/aime25":

                                    if 1: #calculate approximately largest max solution set batch size that fits on GPU (to avoid OOM)
                                        scoring_device_idx = int(hf_devices[0].split(":")[1])
                                        vram,utilisation = gpu_vram_info[scoring_device_idx], (utils.gpu_utilisation(device_idx=scoring_device_idx,all=False))/100
                                        scoring_device_available_vram = vram * (0.9-utilisation) #leave 10% buffer
                                        max_batch_size = utils.max_batch_size_for_memory(scoring_device_available_vram, vocab_size, hidden_size, num_layers, num_attention_heads, token_budget, precision_bytes=2)
                                        solution_set_batch_size = max_batch_size
                                        logger.info(f"solution_set_batch_size: {solution_set_batch_size}")


                                        #for debuging
                                        #mem_regular_fp = utils.regular_forward_pass_memory(vocab_size, hidden_size, num_layers, num_attention_heads, solution_set_batch_size, token_budget, precision_bytes=2).item()
                                        #mem_hidden_state_fp = utils.hidden_states_only_memory(vocab_size, hidden_size, num_layers, num_attention_heads, solution_set_batch_size, token_budget, precision_bytes=2).item()
                                        #mem_regular_fp = mem_hidden_state_fp


                                    utils.clean_up_gpus()
                                    torch.cuda.synchronize()


                                    with utils.gpu_util_manager("Solution set distribution calculation", logger,report_absolutes=True):
                                        with utils.timer("Solution set distribution calculation", logger):
                                            
                                            solution_set_distribution = utils.auto_batch_scale_oom_guard(initial_size=solution_set_batch_size,
                                            operation_func = utils.calculate_solution_set_probs_aime_with_aime_prompt,
                                            logger = logger, 
                                            description = "Solution set distribution calculation",
                                            scale_factor = 0.6, 
                                            min_size = 1, 
                                            #now all args for operation func
                                            reasoning_trace = reasoning_trace,
                                            model = model, 
                                            tokenizer = tokenizer, 
                                            force_end_delim = force_end,
                                            first_token_logits = first_token_logits,
                                            config = config,
                                            lm_head_manual = True,
                                            log_probs = True,
                                            )

                                    #clean up 
                                    torch.cuda.empty_cache()
                                    gc.collect()

                                else: 

                                    solution_set_distribution = [] #to store probs for each candidate solution

                                    batch_reasoning_traces = [reasoning_trace for _ in range(solution_set_batch_size)]
                                    batch_first_token_logits = [first_token_logits for _ in range(solution_set_batch_size)]

                                    for solution_set_batch_idx in range(0,len(solution_set_tokens),solution_set_batch_size):
                                        batch_candidate_solution_tokens = solution_set_tokens[solution_set_batch_idx:solution_set_batch_idx+solution_set_batch_size]

                                        utils.clean_up_gpus()

                                        with utils.gpu_util_manager("Batch calculate answer probability", logger,report_absolutes=True):
                                            with utils.timer("Batch calculate answer probability", logger):
                                                batch_candidate_solution_probs,_ = utils.batch_calculate_answer_probability(
                                                    answer_tokens=batch_candidate_solution_tokens,
                                                    reasoning_traces=batch_reasoning_traces,
                                                    force_end_delim=force_end,
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    config=config,
                                                    first_token_logits=batch_first_token_logits,
                                                    return_entropy=False
                                                )

                                        solution_set_distribution.extend(batch_candidate_solution_probs) #add solution set probs to distribution

                                        #clean up
                                        torch.cuda.empty_cache()
                                        gc.collect()


                                #make solution set distribution safe
                                solution_set_distribution = np.array(solution_set_distribution)
                                ALL_ZEROS_BOOL = np.all(solution_set_distribution == 0)
                                safe_solution_set_dist = solution_set_distribution + 1e-10 #for entropy calculation
                                norm_solution_set_distribution = safe_solution_set_dist / sum(safe_solution_set_dist)

                                #compute metrics from solution set distribution 
                                completion_score = utils.score_answer(model_answer, ground_truth_answer,config)
                                prob_ground_truth = norm_solution_set_distribution[ground_truth_idx].item() if not ALL_ZEROS_BOOL else 0
                                solution_set_entropy = utils.calculate_entropy_via_weighted_surprisal(norm_solution_set_distribution).item()
                                ranking_ground_truth = (np.argsort(norm_solution_set_distribution)[::-1].tolist().index(ground_truth_idx))+1 #1-index the ranking

                                #append to data structure
                                batch_metric_data['answer_score'].append((sample_idx, completion_idx, completion_score))
                                batch_metric_data['answer_probability'].append((sample_idx, completion_idx, prob_ground_truth))
                                batch_metric_data['answer_entropy'].append((sample_idx, completion_idx, solution_set_entropy))
                                batch_metric_data['answer_ranking'].append((sample_idx, completion_idx, ranking_ground_truth))
                                batch_metric_data['solution_set_distribution'].append((sample_idx, completion_idx, norm_solution_set_distribution))
                                
                    if not config.normalise_over_solution_set:

                        for sample_idx in tqdm(range(len(batch_input_tokens)), desc=f"Processing batch samples (budget={token_budget})"):
                        
                            if config.dataset_name=="Idavidrein/gpqa":
                                ground_truth = dataset_samples[sample_idx]['MCQ answer']
                            else:
                                ground_truth = dataset_samples[sample_idx]['answer']

                            ground_truth_tokens = tokenizer(ground_truth.strip(), return_tensors='pt',add_special_tokens=False)['input_ids'][0]

                            for completion_idx in tqdm(range(config.num_completions), desc="Processing completions)",leave=False):

                                output_text = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['text']
                                final_text_output = output_text[-1]
                                reasoning_trace, model_output_raw = ''.join(final_text_output.rsplit(force_end)[:-1]), final_text_output.rsplit(force_end)[-1]
                                model_answer = model_output_raw.split("}")[0].strip()
                                first_token_logits = SAVE_DATA[token_budget][sample_idx]['completions'][completion_idx]['logits'].to('cpu') #we only save first token logits

                                #get answer score

                                completion_score = utils.score_answer(model_answer, ground_truth,config)
                                p,h = utils.calculate_answer_probability(ground_truth_tokens,reasoning_trace,force_end,model,tokenizer,config,first_token_logits,return_entropy=True)
                                prob = p 
                                entropy = h 
                                ranking = None #ranking not defined when not normalising over solution set

                                #append to data structure
                                batch_metric_data['answer_score'].append((sample_idx, completion_idx, completion_score))
                                batch_metric_data['answer_probability'].append((sample_idx, completion_idx, prob))
                                batch_metric_data['answer_entropy'].append((sample_idx, completion_idx, entropy))
                                batch_metric_data['answer_ranking'].append((sample_idx, completion_idx, ranking))
                                batch_metric_data['solution_set_distribution'].append((sample_idx, completion_idx, None))

                    existing_metric_data = SAVE_DATA[token_budget]['metrics']
                    new_metric_data = {k: existing_metric_data[k] + batch_metric_data[k] for k in batch_metric_data} #haven't tested this yet
                    SAVE_DATA[token_budget]['metrics'] = new_metric_data
                    

                    batch_scoring_end_time = time.time()
                    batch_scoring_time = batch_scoring_end_time - batch_scoring_start_time
                    logger.info(f"Scoring for batch {batch_idx + 1}/{len(sample_batches)} took {batch_scoring_time:.2f} seconds")


                    logger.info(f"GPU utilisation after scoring: {utils.gpu_utilisation(all=True)}")

            logger.info(f"Finished generation and scoring for batch {batch_idx + 1}/{len(sample_batches)}")

            #====================================================================================
            #                             SAVING DATA
            #====================================================================================
            if 1: #saving


                if config.SAVE_BOOL:
                    
                    if config.TIME_BASED_SAVING:

                        time_since_last_save_seconds = time.time() - last_save_time
                        time_since_last_save_minutes = int(round(time_since_last_save_seconds / 60))
                        logger.info(f"Time since last save: {time_since_last_save_minutes} minutes")

                        time_based_save_bool = time_since_last_save_minutes > config.save_every_n_mins
                        last_batch_save_bool = batch_idx==len(sample_batches)-1
                        final_save_bool = token_budget.item()==2**config.end_token_budget and last_batch_save_bool

                        if time_based_save_bool: 

                            logger.info(f"Saving at time based interval")
                            
                            utils.save_data(SAVE_DATA, config.save_dir, token_budget=token_budget, batch_idxs=batches_in_save_data, config_save=False)

                            logger.info(f"Saved batches: {batches_in_save_data} to {config.save_dir} at {time.strftime('%Y-%m-%d %H:%M:%S')}")


                            #re-init if we save
                            last_save_time = time.time()
                            batches_in_save_data = []
                            SAVE_DATA={}
                            SAVE_DATA[token_budget]={}
                            SAVE_DATA[token_budget]['metrics'] = {
                                'answer_score': [],
                                'answer_probability': [],
                                'answer_entropy': [],
                                'answer_ranking': [],
                                'solution_set_distribution': [],
                            } #re-init metrics

                        elif last_batch_save_bool: #save at end of token budget

                            logger.info(f"Saving at end of token budget")

                            utils.save_data(SAVE_DATA, config.save_dir, token_budget=token_budget, batch_idxs=batches_in_save_data, config_save=False)
                            logger.info(f"Saved batches: {batches_in_save_data} to {config.save_dir} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                            SAVE_DATA={}
                            SAVE_DATA[token_budget]={}
                            SAVE_DATA[token_budget]['metrics'] = {
                                'answer_score': [],
                                'answer_probability': [],
                                'answer_entropy': [],
                                'answer_ranking': [],
                                'solution_set_distribution': [],
                            } #re-init metrics
                            
                        elif final_save_bool:  #save at end of run

                            logger.info(f"Saving at end of run")

                            if len(batches_in_save_data)>0:
                                utils.save_data(SAVE_DATA, config.save_dir, token_budget=token_budget, batch_idxs=batches_in_save_data, config_save=False)
                                logger.info(f"Saved remaining batches: {batches_in_save_data} to {config.save_dir} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                                SAVE_DATA={}
                                SAVE_DATA[token_budget]={}
                                SAVE_DATA[token_budget]['metrics'] = {
                                    'answer_score': [],
                                    'answer_probability': [],
                                    'answer_entropy': [],
                                    'answer_ranking': [],
                                    'solution_set_distribution': [],
                                } #re-init metrics
                        
                        else: 
                            pass




                    else: #save at end of token budget
                        if batch_idx==len(sample_batches)-1:
                            utils.save_data(SAVE_DATA, config.save_dir, token_budget=token_budget, config_save=False)
                            logger.info(f"Saved SAVE_DATA to {config.save_dir}")
                            SAVE_DATA={} #re-init SAVE_DATA
                            logger.info(f"Saved SAVE_DATA to {config.save_dir} at {time.strftime('%Y-%m-%d %H:%M:%S')}")



        token_budget_end_time = time.time()
        token_budget_time = token_budget_end_time - token_budget_start_time
        logger.info(f"Total time for token budget {token_budget}: {token_budget_time:.2f} seconds")




    print("\n" + "="*50)
    print("✨ FINISHED GENERATION AND SCORING ✨")
    print("="*50 + "\n")



if __name__ == "__main__":
    main()