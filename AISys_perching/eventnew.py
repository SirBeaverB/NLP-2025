from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import sys
from torch.profiler import profile, record_function, ProfilerActivity
import pickle
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os, pickle
from collections import defaultdict


def summarize_profiler_events(prof):
    summary = defaultdict(lambda: {"cpu_time": 0.0, "cuda_time": 0.0})
    for evt in prof.events():
        summary[evt.name]["cpu_time"] += evt.cpu_time_total
        summary[evt.name]["cuda_time"] += evt.cuda_time_total
    return summary

def summarize_by_stage(summary):
    # 先初始化每个阶段
    stage_summary = {
        "Embedding & Input Norm": {"cpu_time": 0.0, "cuda_time": 0.0},
        "Attention QKV+RoPE": {"cpu_time": 0.0, "cuda_time": 0.0},
        "Attention Weight": {"cpu_time": 0.0, "cuda_time": 0.0},
        "Attention Output": {"cpu_time": 0.0, "cuda_time": 0.0},
        "FFN": {"cpu_time": 0.0, "cuda_time": 0.0},
        "Tensor Manipulation": {"cpu_time": 0.0, "cuda_time": 0.0},
        "Memory/Device Ops": {"cpu_time": 0.0, "cuda_time": 0.0},
        "CUDA Kernel/Launch": {"cpu_time": 0.0, "cuda_time": 0.0},
        "Other": {"cpu_time": 0.0, "cuda_time": 0.0}
    }

    for name, times in summary.items():
        cpu_time = times["cpu_time"]
        cuda_time = times["cuda_time"]
        # Embedding & Input Norm
        if "aten::embedding" in name or "aten::layer_norm" in name:
            stage_summary["Embedding & Input Norm"]["cpu_time"] += cpu_time
            stage_summary["Embedding & Input Norm"]["cuda_time"] += cuda_time
        # QKV+RoPE
        elif "rotary" in name or "rope" in name:
            stage_summary["Attention QKV+RoPE"]["cpu_time"] += cpu_time
            stage_summary["Attention QKV+RoPE"]["cuda_time"] += cuda_time
        # Attention 权重
        elif (
            "aten::matmul" in name or
            "aten::bmm" in name or
            "aten::softmax" in name or
            "aten::dropout" in name or
            "aten::_flash_attention_forward" in name or
            "aten::_scaled_dot_product_flash_attention" in name or
            "scaled_dot_product_attention" in name
        ):
            stage_summary["Attention Weight"]["cpu_time"] += cpu_time
            stage_summary["Attention Weight"]["cuda_time"] += cuda_time
        # Attention 输出
        elif "aten::add" in name:
            stage_summary["Attention Output"]["cpu_time"] += cpu_time
            stage_summary["Attention Output"]["cuda_time"] += cuda_time
        # FFN
        elif "aten::gelu" in name or "aten::relu" in name or "aten::silu" in name:
            stage_summary["FFN"]["cpu_time"] += cpu_time
            stage_summary["FFN"]["cuda_time"] += cuda_time
        # linear 平分
        elif "aten::linear" in name:
            for stage in ["Attention QKV+RoPE", "Attention Output", "FFN"]:
                stage_summary[stage]["cpu_time"] += cpu_time / 3
                stage_summary[stage]["cuda_time"] += cuda_time / 3
        # Tensor Manipulation
        elif (
            "aten::cat" in name or "aten::view" in name or "aten::reshape" in name or
            "aten::contiguous" in name or "aten::unsqueeze" in name or "aten::squeeze" in name or
            "aten::transpose" in name or "aten::slice" in name or "aten::select" in name or
            "aten::expand" in name or "aten::index" in name or "aten::unbind" in name
        ):
            stage_summary["Tensor Manipulation"]["cpu_time"] += cpu_time
            stage_summary["Tensor Manipulation"]["cuda_time"] += cuda_time
        # Memory/Device Ops
        elif (
            "aten::empty" in name or "aten::empty_like" in name or "aten::clone" in name or
            "aten::copy_" in name or "aten::to" in name or "cudaMalloc" in name or
            "cudaMemcpy" in name or "cudaFree" in name or "cudaHostAlloc" in name or
            "cudaMemset" in name or "cudaMemsetAsync" in name or "cudaStreamSynchronize" in name or
            "cudaLaunchKernel" in name or "cudaDeviceSynchronize" in name or "cudaFuncGetAttributes" in name
        ):
            stage_summary["Memory/Device Ops"]["cpu_time"] += cpu_time
            stage_summary["Memory/Device Ops"]["cuda_time"] += cuda_time
        # CUDA Kernel/Launch
        elif (
            "cudaLaunchKernel" in name or "cudaFuncSetAttribute" in name or "cudaOccupancy" in name or
            "ampere_bf16" in name or "pytorch_flash" in name or "gemmk1_kernel" in name or
            "void at::native" in name
        ):
            stage_summary["CUDA Kernel/Launch"]["cpu_time"] += cpu_time
            stage_summary["CUDA Kernel/Launch"]["cuda_time"] += cuda_time
        else:
            stage_summary["Other"]["cpu_time"] += cpu_time
            stage_summary["Other"]["cuda_time"] += cuda_time
    return stage_summary

def run_llm(batchsize, prompt_len):
    # 加载模型
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        config=config
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 构造输入
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

    # 配置 profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        profile_memory=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:

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
                
    # 保存结果文件夹
    results_dir = "profiler_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存原始 profiler 数据
    raw_file = os.path.join(results_dir, f"test_b{batchsize}_p{prompt_len}.pkl")
    with open(raw_file, 'wb') as f:
        pickle.dump(prof.profiler.function_events, f)

    # 保存汇总事件
    summary = summarize_profiler_events(prof)
    summary_file = os.path.join(results_dir, f"test_b{batchsize}_p{prompt_len}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Aggregated Module Runtime Summary ===\n")
        for name in sorted(summary):
            cpu_time = summary[name]["cpu_time"]
            cuda_time = summary[name]["cuda_time"]
            f.write(f"{name:<40} | CPU: {cpu_time:.2f} us | CUDA: {cuda_time:.2f} us\n")

    # 新增：写入各阶段总耗时
    stage_summary = summarize_by_stage(summary)
    stage_summary_file = os.path.join(results_dir, f"test_b{batchsize}_p{prompt_len}_stage_summary.txt")
    with open(stage_summary_file, 'w') as f:
        f.write("=== Stage-wise Runtime Summary ===\n")
        for stage, times in stage_summary.items():
            f.write(f"{stage:<25} | CPU: {times['cpu_time']:.2f} us | CUDA: {times['cuda_time']:.2f} us\n")

    return summary


def run_batch_size_tests():
    batch_sizes = [1, 2, 4, 8, 16]
    fixed_prompt_len = 200  # 固定prompt长度
    
    for batch_size in batch_sizes:
        print(f"\nRunning test with batch_size={batch_size}, prompt_len={fixed_prompt_len}")
        run_llm(batch_size, fixed_prompt_len)
        print(f"Completed test for batch_size={batch_size}")

def run_prompt_length_tests():
    prompt_lengths = [50, 100, 200, 400, 800, 1600]
    fixed_batch_size = 4  # 固定batch size
    
    for prompt_len in prompt_lengths:
        print(f"\nRunning test with batch_size={fixed_batch_size}, prompt_len={prompt_len}")
        run_llm(fixed_batch_size, prompt_len)
        print(f"Completed test for prompt_len={prompt_len}")

if __name__ == "__main__":
    # 运行batch size测试
    print("Running batch size tests...")
    run_batch_size_tests()
    
    # 运行prompt length测试
    print("\nRunning prompt length tests...")
    run_prompt_length_tests()
