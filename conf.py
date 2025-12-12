import torch, numpy as np, os, psutil, argparse, sys, yaml
from datetime import datetime
from cli_args import has_script_arguments
from dotenv import load_dotenv
from ast import literal_eval
import utils.tokenization_setup as tokenization_setup

class Config:

    def __init__(self,parse_args=True):

        """
        Priority order: base config < yaml config < CLI overrides

        """
        
        #set base config (config below)
        self._set_base_config()
        
        # parse arguments, including config file
        if parse_args and len(sys.argv) > 1:
            self._parse_arguments()

            #load yaml config
            self._load_yaml_config()

            #override yaml with cli inputs - we don't really use this anymore
            self._apply_cli_overrides()


    
        
        # Compute derived attributes
        self._set_derived_attributes()

    def _parse_arguments(self):
        """Parse both positional and optional arguments"""
        parser = argparse.ArgumentParser()
        
        # optional argument for config file
        parser.add_argument("--config_file",type=str,default=None,help="Path to YAML config file")

        
        # Optional arguments that can override config
        parser.add_argument("--model_name", type=str, default=None)
        parser.add_argument("--dataset_name", type=str, default=None)
        parser.add_argument("--start_token_budget", type=int, default=None)
        parser.add_argument("--end_token_budget", type=int, default=None)
        parser.add_argument("--num_samples", type=str, default=None)
        parser.add_argument("--sample_idxs_range", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--inference_engine", type=str, default=None)
        parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=None)
        parser.add_argument("--num_completions", type=int, default=None)
        parser.add_argument("--quantization",type=int,default=None)
        parser.add_argument("--force_end_batch_size",type=int,default=None)
        parser.add_argument("--solution_set_batch_size", type=int, default=None)
        parser.add_argument("--normalise_over_solution_set", action="store_true", default=False)
        parser.add_argument("--max_new_tokens_frac", type=float, default=None)
        parser.add_argument("--SAVE_BOOL", action="store_true", default=False)
        parser.add_argument("--ATTENTION_TYPE", type=str, default=None)
        self.args, self.unknown = parser.parse_known_args()
        
        # Set config file from positional argument
        self.config_file = self.args.config_file

    def _apply_cli_overrides(self):
        """Apply CLI arguments as overrides after YAML loading"""
        if hasattr(self, 'args'):
            # Apply each non-None argument as override

            if self.args.config_file is not None:
                return #if config file, don't do any CLI overrides

            #specials 
            if self.args.sample_idxs_range is not None:
                if self.args.sample_idxs_range=="None":
                    self.sample_idxs_range = None
                else:
                    self.sample_idxs_range = literal_eval(self.args.sample_idxs_range)
            else:
                self.sample_idxs_range = None

            if self.args.num_samples is not None:
                if self.args.num_samples=="None":
                    self.num_samples = None
                else:
                    self.num_samples = int(self.args.num_samples)
            else: 
                self.num_samples = None
                

            if self.args.model_name is not None:
                self.model_name = self.args.model_name
            if self.args.dataset_name is not None:
                self.dataset_name = self.args.dataset_name
            if self.args.start_token_budget is not None:
                self.start_token_budget = self.args.start_token_budget
            if self.args.end_token_budget is not None:
                self.end_token_budget = self.args.end_token_budget
            if self.args.batch_size is not None:
                self.batch_size = self.args.batch_size
            if self.args.inference_engine is not None:
                self.inference_engine = self.args.inference_engine
            if self.args.vllm_gpu_memory_utilization is not None:
                self.vllm_gpu_memory_utilization = self.args.vllm_gpu_memory_utilization
            if self.args.num_completions is not None:
                self.num_completions = self.args.num_completions
            if self.args.max_new_tokens_frac is not None:
                self.max_new_tokens_frac = self.args.max_new_tokens_frac
            if self.args.quantization is not None:
                self.quantization = self.args.quantization
            if self.args.force_end_batch_size is not None:
                self.force_end_batch_size = self.args.force_end_batch_size
            if self.args.normalise_over_solution_set:
                self.normalise_over_solution_set = True
            if self.args.SAVE_BOOL:
                self.SAVE_BOOL = True
            if self.args.solution_set_batch_size is not None:
                self.solution_set_batch_size = self.args.solution_set_batch_size
            if self.args.ATTENTION_TYPE is not None:
                self.ATTENTION_TYPE = self.args.ATTENTION_TYPE
                

    def _set_base_config(self):
        
        #workflow scale parameters 
        self.num_samples = None
        self.sample_idxs_range = None
        self.batch_size = 4
        self.gpu_memory_utilization = 0.4
        self.max_model_len = None
        self.num_completions = 1
        self.USE_TOKEN_BUDGET_BATCH_SIZE_MAPPING = False
        self.token_batch_mapping = {
            "default": (4,4),
            "4096": (2,10),
            "8192": (2,10),
        }

        
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer_name = None
        self.dataset_name = "math-ai/aime25"
        self.SAVE_BOOL = False
        self.start_token_budget, self.end_token_budget = 8, 11
        self.DEBUG_MODE = False
        self.inference_engine = "hf" 
        if self.inference_engine == "vllm": 
            assert torch.cuda.is_available(), "VLLM requires CUDA"
        self.vllm_gpu_memory_utilization = 0.4
        self.vllm_max_num_seqs = 64


        self.quantization = None

        # Model generation config
        self.max_new_tokens_frac = float(1/8)
        self.final_answer_budget = 10
        self.do_sample = True
        self.default_temperature = 0.7
        self.default_top_p = 0.95
        self.default_top_k = 20
        self.default_repetition_penalty = 1.1
        self.return_dict_in_generate = True
        self.output_scores = True
        self.output_hidden_states = False

        # general workflow config
        self.force_end = "FINAL ANSWER: \\boxed{"
        self.safety_buffer = 50 #for creating new sequences tensor
        self.answer_indicator_token = "boxed"
        self.USE_LOCAL_HF_CACHE = None
        self.ATTENTION_TYPE = "mem_efficient" # "flash", "mem_efficient", "vanilla"
        self.bad_words = ["dummy"]
    
        #force continue config
        self.force_continue = " Hmm, but let me keep thinking... <think>"
        self.REMOVE_ANSWER_AT_CONTINUATION = False
        self.n_tokens_remove_answer = 10
        self.INSERT_SYSTEM_PROMPT_FOR_FORCE_CONTINUE = False
        self.force_continue_repetition_threshold = 100 #
        self.CHECK_FOR_FORCE_CONTINUE_REPETITIONS = True
        self.APPEND_EOS_TOKEN = True #This should usually be false, just for debugging. 
        
        #dataset config 
        self.aime_answer_prompt = True
        self.INSERT_SYSTEM_PROMPT_FOR_GPQA_MCQ_FORMAT = False

        # scoring config 
        self.normalise_over_solution_set = False
        self.LOCALISATION_RUN = False

    
        #save config
        self.TIME_BASED_SAVING = True #if false, saves after all batches done - i.e: every token budget 
        self.save_every_n_mins = 120 #save every 2 hours


        #batch sizes
        self.force_end_batch_size = None
        self.solution_set_batch_size = 50
        self.calculate_solution_set_batch_size = False



        self.config_file = None

    def _load_yaml_config(self):

        """Load configuration from YAML file"""
        if self.config_file is None:
            return
            
        print(f"DEBUG: loading yaml config {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Apply each setting from the YAML file
            for key, value in config_data.items():
                if value == 'None':
                    value = None
                setattr(self, key, value)
            
            
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")


            

    def _set_derived_attributes(self):

        # end reasonning trace properly 
        self.force_end = "</think>" + self.force_end


        # INSERT_YOUR_CODE
        if self.num_samples is not None and self.sample_idxs_range is not None:
            raise ValueError("num_samples and sample_idxs_range cannot both be set. Please specify only one.")
        if self.sample_idxs_range is not None:
            self.num_samples = self.sample_idxs_range[1] - self.sample_idxs_range[0]
        
        if self.num_samples == "None":
            self.num_samples = None
        if self.num_samples is not None:
            self.num_samples = int(self.num_samples)
        

        if self.tokenizer_name is None: 
            self.tokenizer_name = self.model_name

        #batch size setting 
        if self.force_end_batch_size is None:
            self.force_end_batch_size = self.batch_size * self.num_completions
        else:
            self.force_end_batch_size = self.num_completions #minimal - hopefully this doesn't increase time cmoplexity significantly. 

        assert self.ATTENTION_TYPE in ["flash", "mem_efficient", "vanilla"], f"Attention type {self.ATTENTION_TYPE} not in ['flash', 'mem_efficient', 'vanilla']"

        # Dataset checks
        dataset_name_all = [
            "openai/gsm8k",
            "math-ai/aime25",
            "MathArena/hmmt_feb_2025",
            "Idavidrein/gpqa",
            "Maxwell-Jia/AIME_2024",
            "ProCreations/SimpleMath",
        ]
        assert self.dataset_name in dataset_name_all, f"Dataset {self.dataset_name} not in {dataset_name_all}"

        #set end of input tokens 
        if "deepseek" in self.model_name:
            self.end_of_input_tokens = tokenization_setup.deepseek_end_of_input_ids
        elif "QwQ" in self.model_name:
            self.end_of_input_tokens = tokenization_setup.qwq_end_of_input_ids
        elif "Phi" in self.model_name:
            self.end_of_input_tokens = tokenization_setup.phi_end_of_input_ids
        elif "Qwen" in self.model_name:
            self.end_of_input_tokens = tokenization_setup.qwen_end_of_input_ids
        elif "gemma-2" in self.model_name:
            self.end_of_input_tokens = tokenization_setup.gemma_end_of_input_ids
        else:
            raise ValueError(f"Model {self.model_name} not supported")

        # Token budgets
        if self.LOCALISATION_RUN:
            token_budgets_all = []
            self.start_token_budget = np.array(self.start_token_budget)
            self.end_token_budget = self.start_token_budget + 1


            assert len(self.start_token_budget) == len(self.sample_idxs_range), "start_token_budget and sample_idxs_range must have the same length"

            for idx,(st,end) in enumerate(list(zip(self.start_token_budget,self.end_token_budget))):
                fractions = np.arange(0,1,self.max_new_tokens_frac)
                budgets = [int(2**st + f * (2**end - 2**st)) for f in fractions]
                budgets.append(int(2**end))
                budgets = [int(t) for t in budgets]
                token_budgets_all.append(budgets)

            self.token_budgets = token_budgets_all

        else: 
            self.token_budgets = list(np.logspace(
                start=self.start_token_budget, 
                stop=self.end_token_budget, 
                base=2, 
                num=self.end_token_budget - self.start_token_budget + 1
            ).astype(int))

        

        if self.max_model_len is None:
            if self.LOCALISATION_RUN:
                self.max_model_len = max([max(budgets) for budgets in self.token_budgets]).item()
            else:
                self.max_model_len = self.token_budgets[-1].item()

            if "gpqa" in self.dataset_name.lower():
                max_model_len_buffer = max(2048,int(0.25*self.max_model_len)) + 3000 #to catch oversamplings
            else: 
                max_model_len_buffer = max(2048,int(0.25*self.max_model_len))
            self.max_model_len = int(self.max_model_len) + max_model_len_buffer #add 1/4 of token budget for safety buffer
                
        if self.TIME_BASED_SAVING:
            assert self.save_every_n_mins is not None, "save_every_n_mins must be specified for time-based saving"
        
        # Save directory
        remote_disk_dir = os.getenv("scratch_disk_dir")

        # Save directory
        if os.environ.get("node_type") == "EIDF":
            self.data_dir = "/outputs/results_data"
        else:
            self.data_dir = "./results_data"

        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        self.run_name = f"{self.model_name.split('/')[-1]}_{self.dataset_name.split('/')[-1]}_{datetime.now().strftime('%m-%d_%H-%M-%S')}"
        self.save_dir = os.path.join(self.data_dir, self.run_name)

        if self.LOCALISATION_RUN:
            self.save_dir = self.save_dir + "_localisation_run"

        os.makedirs(self.save_dir, exist_ok=True)


        #scoring 
        if self.dataset_name=="Idavidrein/gpqa":
            self.solution_set_batch_size = 4 # bit hacky but that's okay for now.

        #localisation checks 
        if self.LOCALISATION_RUN:
            assert self.num_samples is None, "Localisation run requires num_samples to be None"
            assert self.sample_idxs_range is not None, "Localisation run requires sample_idxs_range to be specified"


    def __str__(self):
        """Pretty print config details in organized sections"""
        sections = {
            "Model Configuration": [
                ("Model Name", self.model_name),
                ("Tokenizer Name", self.tokenizer_name),
                ("Inference Engine", self.inference_engine),
                ("Quantization", self.quantization),
                ("GPU Memory Utilization", self.vllm_gpu_memory_utilization if hasattr(self, 'vllm_gpu_memory_utilization') else 'N/A')
            ],
            "Dataset Configuration": [
                ("Dataset Name", self.dataset_name),
                ("Number of Samples", self.num_samples),
                ("Sample Indices Range", self.sample_idxs_range),
                ("Number of Completions", self.num_completions),
                ("Batch Size", self.batch_size)
            ],
            "Token Budget": [
                ("Start Token Budget", self.start_token_budget),
                ("End Token Budget", self.end_token_budget),
                ("Token Budgets", self.token_budgets),
                ("Use Budget Batch Mapping", self.USE_TOKEN_BUDGET_BATCH_SIZE_MAPPING),
            ],
            "Generation Configuration": [
                ("Max New Tokens Fraction", self.max_new_tokens_frac),
                ("Final Answer Budget", self.final_answer_budget),
                ("Force End Batch Size", self.force_end_batch_size),
                ("Do Sample", self.do_sample),
                ("Max Model Length", self.max_model_len),
                ("Attention Type", self.ATTENTION_TYPE),
            ],
            "Localisation Settings": [
                ("Localisation Run", self.LOCALISATION_RUN),
                ("Localisation Sample Indices", self.sample_idxs_range if self.LOCALISATION_RUN else 'N/A')
            ],
            "Scoring Configuration": [ 
                ("Normalise Over Solution Set", self.normalise_over_solution_set),
                ("Solution Set Batch Size", self.solution_set_batch_size)
            ],
            "Save Configuration": [
                ("Save Enabled", self.SAVE_BOOL),
                ("Save Directory", self.save_dir),
                ("Run Name", self.run_name)
            ]
        }
        
        output = []
        for section, items in sections.items():
            output.append(f"\n{section}:")
            output.extend(f"  {key}: {value}" for key, value in items)
        
        return "\n".join(output)

    def __repr__(self):
        return self.__str__()



#works with jupyter or cli arg runs
#can drop this because we're not using jupyter nb anymore for main script
config = Config(parse_args=has_script_arguments())