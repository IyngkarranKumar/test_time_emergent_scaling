

class TraceConfig:

    def __init__(self):
        
        self.DEBUG_MODE = True
        self.API = "openrouter"
        self.RUN_API=False
        if self.DEBUG_MODE:
            self.RUN_API=False

        self.tag_model = "deepseek/deepseek-r1-distill-llama-70b:free"
        self.system_prompt = "trace_analysis/trace_prompts/system_prompt.txt"
        self.core_prompt = "trace_analysis/trace_prompts/core.txt"

        self.prompt_text_paths = {
            "core": "trace_analysis/trace_prompts/core.txt",
            "main_example_1": "trace_analysis/trace_prompts/example_1_thoughtology.txt",
            "repetition_example_1": "trace_analysis/trace_prompts/example_4_repetitive.txt",
        }

        if self.DEBUG_MODE:
            self.prompt_text_paths["main_example_2"] = None
            self.prompt_text_paths["main_example_3"] = None  
            self.prompt_text_paths["repetition_example_2"] = None

        else:
            self.prompt_text_paths["main_example_2"] = "trace_analysis/trace_prompts/example_2_AIME.txt"
            self.prompt_text_paths["main_example_3"] = "trace_analysis/trace_prompts/example_3_GPQA.txt"
            self.prompt_text_paths["repetition_example_2"] = "trace_analysis/trace_prompts/example_5_repetitive.txt"


