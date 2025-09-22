import numpy as np 
import torch
import warnings
import random
import ast

from datasets import load_dataset
from utils.tokenization_setup import qwen_end_of_input_ids, gemma_end_of_input_ids, deepseek_end_of_input_ids, qwq_end_of_input_ids, phi_end_of_input_ids

np.random.seed(42)
random.seed(42)

GPQA_MCQ_FORMAT="Choose from one of the four options below. Give your answer as a single letter: A, B, C, or D. The only answer you can submit is one of A, B, C, or D. Do not submit anything else. "

AIME_ANSWER_PROMPT="The answer is an integer between 0 and 999 inclusive. Give all answers as 3 digit numbers. For example, if the answer is 7 your final answer should be 007. If the answer is 45 your final answer should be 045. If the answer is 123 your final answer should be 123."

MATH_PROMPT = "\n\n Please reason step by step, and put your final answer within \\boxed{}\n."

SYSTEM_PROMPT_FOR_FORCE_CONTINUE = lambda force_continue: f""" When you see {{{force_continue}}}, rigorously verify your answer by:


Numerical verification:
- Redo all calculations step-by-step with explicit arithmetic
- Double-check unit conversions and orders of magnitude
- Verify any formulas used

Alternative method:
- Solve using a completely different approach if possible
- Work backwards from your answer to verify it fits the original problem
- Check your reasoning against physical intuition/common sense

Option analysis:
- Systematically compare your answer against each given option
- Explain why the other options are incorrect
- Confirm your choice makes sense in context

Be willing to change your answer if you think you've made a mistake. 

"""

SYSTEM_PROMPT_GPQA_MCQ_ANSWER = "\n\n This task is a multiple choice question, so your final answer must be one of the four following options: A, B, C, or D. Nothing else."


#i think we don't want the model seeing the force end prompt until prompted
#FORCE_END_PROMPT_EDIT = lambda force_end_delimiter: f"\nWhen you see: {force_end_delimiter}, give your final answer directly. \n"

#dataset utils
def assert_padding_consistency(input_ids, attention_mask,tokenizer,elementwise=False):

    if not elementwise:
        """Assert that the number of pad tokens equals the number of 0s in attention mask."""
        num_pad_tokens = (input_ids == tokenizer.pad_token_id).sum().item()
        num_attention_zeros = (attention_mask == 0).sum().item()
        assert num_pad_tokens == num_attention_zeros, f"Padding mismatch: {num_pad_tokens} pad tokens but {num_attention_zeros} attention zeros"
    if elementwise:
        """Assert that each pad token corresponds to a 0 in attention mask."""
        pad_token_positions = (input_ids == tokenizer.pad_token_id)
        attention_zero_positions = (attention_mask == 0)
        assert torch.equal(pad_token_positions, attention_zero_positions), "Padding positions do not match attention mask zeros"

    
def prepare_batch_inputs(batch_texts,tokenizer,config,device,with_chat_template=True,force_end_prompt=None):

    """Prepare batch inputs with padding"""

    #to try to get model to give answer after seeing force-end delimiter
    
    batch_texts = [text + MATH_PROMPT for text in batch_texts]

    if with_chat_template:
        messages = [[
            {"role": "user", "content": text}] 
            for text in batch_texts]


        if tokenizer.chat_template is not None:
            batch_texts = [tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages] #add geneation prompt needed for starting generation 
            
            if config.INSERT_SYSTEM_PROMPT_FOR_FORCE_CONTINUE:


                if "Qwen2.5" in config.model_name:
                    end_of_input_tokens = tokenizer.decode(qwen_end_of_input_ids,skip_special_tokens=False)
                elif "gemma-2" in config.model_name:
                    end_of_input_tokens = tokenizer.decode(gemma_end_of_input_ids,skip_special_tokens=False)
                elif "DeepSeek" in config.model_name:
                    end_of_input_tokens = tokenizer.decode(deepseek_end_of_input_ids,skip_special_tokens=False)
                elif "QwQ" in config.model_name:
                    end_of_input_tokens = tokenizer.decode(qwq_end_of_input_ids,skip_special_tokens=False)
                elif "Phi" in config.model_name:
                    end_of_input_tokens = tokenizer.decode(phi_end_of_input_ids,skip_special_tokens=False)
                else:
                    raise ValueError(f"Model {config.model_name} not supported")


                if config.INSERT_SYSTEM_PROMPT_FOR_GPQA_MCQ_FORMAT and config.dataset_name == "Idavidrein/gpqa":
                    batch_texts = [text.replace(end_of_input_tokens, SYSTEM_PROMPT_FOR_FORCE_CONTINUE(config.force_continue) + SYSTEM_PROMPT_GPQA_MCQ_ANSWER + end_of_input_tokens,1) for text in batch_texts]
                
                else:
                    batch_texts = [text.replace(end_of_input_tokens, SYSTEM_PROMPT_FOR_FORCE_CONTINUE(config.force_continue) + end_of_input_tokens,1) for text in batch_texts]                     
        else:
            batch_texts = [msg[0]["content"] for msg in messages]
            batch_texts = [msg[0]["content"] for msg in messages]

    batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, padding_side='left')
    
    # Get lengths of each input prompt
    if tokenizer.chat_template is not None:
        question_lengths_tokens = (batch_inputs['input_ids'] != tokenizer.pad_token_id).sum(dim=1)
    else:
        if tokenizer.bos_token_id is not None:
            question_lengths_tokens = (batch_inputs['input_ids'] != tokenizer.pad_token_id).sum(dim=1) - 1
        else:
            question_lengths_tokens = (batch_inputs['input_ids'] != tokenizer.pad_token_id).sum(dim=1)
            
    #batch_inputs['question_lengths'] = question_lengths_tokens
    
    return batch_inputs.to(device), question_lengths_tokens

def create_batches(samples, batch_size):
    """Create batches from samples"""
    batches = []

    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    return batches


class DatasetSetup:

    def __init__(self, dataset_name,num_samples,config):
        self.dataset_name=dataset_name
        self.num_samples=num_samples

        if dataset_name=="openai/gsm8k":
            self.dataset_config="main"
            self.dataset_split="train"
        elif dataset_name=="math-ai/aime25":
            self.dataset_config=None
            self.dataset_split="test"
            self.solution_set = np.arange(0,1000)
        elif dataset_name=="MathArena/hmmt_feb_2025":
            self.dataset_config=None
            self.dataset_split="train"
        elif dataset_name=="Idavidrein/gpqa":
            self.dataset_config="gpqa_diamond"
            self.dataset_split="train"
            self.dataset_format="mcq" #or qa
            self.solution_set = ["A","B","C","D"]
        elif dataset_name=="Maxwell-Jia/AIME_2024":
            self.dataset_config=None
            self.dataset_split="train"
            self.solution_set = np.arange(0,1000)
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

        self.config = config


        

    def _load_dataset(self): 
        """
        Loads dataset from HuggingFace
        If num_samples is None, loads entire dataset
        If num_samples is not None, loads first 'num_samples' samples
        """
        if self.num_samples is None:
            ds=load_dataset(self.dataset_name,self.dataset_config)[self.dataset_split]
        else:
            ds=load_dataset(self.dataset_name,self.dataset_config)[self.dataset_split].select(range(self.num_samples))

        self.dataset=ds

    def format_sample(self,sample):
        """
        Standardizes sample into q-a format
        """

        if self.dataset_name == "openai/gsm8k":
            return {
                'question': sample['question'],
                'answer_with_trace': sample['answer'],
                'answer':sample['answer'].split("####")[1].strip()
            }

        elif self.dataset_name == "Idavidrein/gpqa":
            question=sample["Pre-Revision Question"]
            answer=sample["Pre-Revision Correct Answer"]
            incorrect_answer_1=sample["Pre-Revision Incorrect Answer 1"]
            incorrect_answer_2=sample["Pre-Revision Incorrect Answer 2"]
            incorrect_answer_3=sample["Pre-Revision Incorrect Answer 3"]

            # Randomly assign answer to A, B, C, or D
            choices = [answer, incorrect_answer_1, incorrect_answer_2, incorrect_answer_3]
            np.random.shuffle(choices)
            
            choice_letters = ['A', 'B', 'C', 'D']
            mcq_answer = choice_letters[choices.index(answer)]
            
            formatted_question = f"{question}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"

            if self.dataset_format=="mcq":
                
                formatted_question=f"{question}\n\n{GPQA_MCQ_FORMAT}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"


            return {
                "question": formatted_question,
                "answer": answer, 
                "MCQ answer": mcq_answer,
            }

        elif self.dataset_name == "math-ai/aime25":
            if self.config.aime_answer_prompt:
                formatted_question=f"{sample['problem']}\n{AIME_ANSWER_PROMPT}"
            else:
                formatted_question=f"{sample['problem']}"
            return {
                'question': formatted_question,
                'answer': sample['answer']
            }
        
        elif self.dataset_name == "Maxwell-Jia/AIME_2024":
            if self.config.aime_answer_prompt:
                formatted_question=f"{sample['Problem']}\n{AIME_ANSWER_PROMPT}"
            else:
                formatted_question=f"{sample['Problem']}"
            return {
                'question': formatted_question,
                'answer': str(sample['Answer'])
            }

        else:
            warnings.warn(f"Not implemented for {self.dataset_name}")
        
    def load_standardized_dataset(self,config):

        self._load_dataset()
        self.dataset = [{'sample_idx': idx, **self.format_sample(sample)} for idx, sample in enumerate(self.dataset)]

        if config.LOCALISATION_RUN:
            self.dataset = [sample for sample in self.dataset if sample['sample_idx'] in config.sample_idxs]
        else:
            if hasattr(config, 'sample_idxs') and config.sample_idxs is not None:
                self.dataset = [sample for sample in self.dataset if sample['sample_idx'] in config.sample_idxs]
        
    

        return self.dataset







