import gc
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_chat_prompt(tokenizer, user_prompt, SYS_PROMPT, molecule_smiles=None):

    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt.replace('{target_smiles}', molecule_smiles),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# Cache the engine so repeated calls to generate() don't reload the model each time
_LLM_CACHE = {}


def _get_llm(model_name):
    if model_name not in _LLM_CACHE:
        # vLLM holds GPU memory for the engine's lifetime; keep only one model loaded
        for old_name in list(_LLM_CACHE.keys()):
            del _LLM_CACHE[old_name]
        gc.collect()
        torch.cuda.empty_cache()

        _LLM_CACHE[model_name] = LLM(
            model=model_name,
            dtype="float16",
            trust_remote_code=True,
            # gpu_memory_utilization=0.90,   # lower this if you hit OOM
            # max_model_len=4096,            # cap context length to save memory
        )
    return _LLM_CACHE[model_name]


def generate(model_name, smiles, user_prompt, top_p, max_tokens, SYS_PROMPT, temperature):

    llm = _get_llm(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    prompt = build_chat_prompt(tokenizer, user_prompt, SYS_PROMPT, smiles)

    # Original used do_sample=False (greedy decoding), so temperature/top_p had no effect.
    # In vLLM, temperature=0 is greedy, which matches that behavior.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    # The chat template already adds special tokens (e.g. BOS), so the prompt is passed as-is
    outputs = llm.generate([prompt], sampling_params)

    return outputs[0].outputs[0].text