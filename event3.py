from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import sys
from torch.profiler import profile, record_function, ProfilerActivity
import pickle

def run_llm(batchsize, prompt_len):  # 参数可以自己设计，可以考虑加入max_new_tokens
    config = AutoConfig.from_pretrained("/public/SothisAI/learning_center/Qwen")
    # config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        "/public/SothisAI/learning_center/Qwen",
        torch_dtype=torch.bfloat16,
        config=config
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        "/public/SothisAI/learning_center/Qwen", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    prompt = (
        "A breathtaking scene from Genshin Impact: a lone traveler "
        "stands atop the high cliffs of Liyue, overlooking the glowing city below at "
        "twilight. Lanterns float into the sky, casting a warm golden light across the "
        "clouds. The wind gently rustles through the traveler's cloak as they prepare "
        "to glide down, sword at their side, guided by destiny and the stars."
    ) * 10
    tokens = tokenizer.encode(prompt)
    tokens = tokens[:int(prompt_len)]
    prompt = tokenizer.decode(tokens)
    prompts = [prompt for _ in range(batchsize)]
    inputs = tokenizer(prompts, add_special_tokens=False, padding=True,
                      truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].cuda()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_stack=True, profile_memory=True,
                 experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        output = model.generate(input_ids, attention_mask=attention_mask,
                               max_new_tokens=32, do_sample=False)
        out_str = tokenizer.batch_decode(output[:, input_ids.size(1):],
                                        skip_special_tokens=True)
        with open(f"./tests_{batchsize}_{prompt_len}.pkl", 'wb') as f:
            pickle.dump(prof.profiler.function_events, f)
        t = 0
        for event in prof.events():
            print(f"name :{event.name}")
            print(f"stack :{event.stack}")
            t = t + 1
            if t == 10:
                break
    return prof.events()