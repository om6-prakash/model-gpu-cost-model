import argparse
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-g', '--num_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('-p', '--prompt_sz', type=int, default=4096, help='Prompt size in tokens')
    parser.add_argument('-r', '--response_sz', type=int, default=256, help='Response size in tokens')
    parser.add_argument('-c', '--n_concurrent_req', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('-w', '-cw', '--ctx_window', type=int, default=1024, help='Average context window')

    args = parser.parse_args()

    num_gpu = args.num_gpu
    prompt_size = args.prompt_sz
    response_size = args.response_sz
    n_concurrent_request = args.n_concurrent_req
    avg_context_window = args.ctx_window

    # Print input
    print(f" num_gpu = {num_gpu}, prompt_size = {prompt_size} tokens, response_size = {response_size} tokens")
    print(f" n_concurrent_request = {n_concurrent_request}, avg_context_window = {avg_context_window} tokens")

    # Define variables
    gpu_specs = [
        {"name": "A10", "fp16_tflops": 125, "memory_gb": 24, "memory_bandwidth_gbps": 600},
        {"name": "A30", "fp16_tflops": 330, "memory_gb": 24, "memory_bandwidth_gbps": 933},
        {"name": "L40", "fp16_tflops": 181, "memory_gb": 48, "memory_bandwidth_gbps": 864},
        {"name": "L40s", "fp16_tflops": 362, "memory_gb": 48, "memory_bandwidth_gbps": 864},
        {"name": "A100 40 GB", "fp16_tflops": 312, "memory_gb": 40, "memory_bandwidth_gbps": 1555},
        {"name": "A100 40 GB SXM", "fp16_tflops": 312, "memory_gb": 40, "memory_bandwidth_gbps": 1555},
        {"name": "A100 80 GB PCIe", "fp16_tflops": 312, "memory_gb": 80, "memory_bandwidth_gbps": 1935},
        {"name": "A100 80 GB SXM", "fp16_tflops": 312, "memory_gb": 80, "memory_bandwidth_gbps": 2039},
        {"name": "H100 PCIe", "fp16_tflops": 1513, "memory_gb": 80, "memory_bandwidth_gbps": 2000},
        {"name": "H100 SXM", "fp16_tflops": 1979, "memory_gb": 80, "memory_bandwidth_gbps": 3350},
        {"name": "H100 NVL", "fp16_tflops": 3958, "memory_gb": 188, "memory_bandwidth_gbps": 7800}
        # Add or comment out GPU types as needed
    ]

    model_specs = [
        {"name": "Llama-3-8B", "params_billion": 8, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 8192, "d_head": 128},
        {"name": "Llama-3-70B", "params_billion": 70, "d_model": 8192, "n_heads": 64, "n_layers": 80, "max_context_window": 8192, "d_head": 128},
        {"name": "Llama-3.1-8B", "params_billion": 8, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 131072, "d_head": 128},
        {"name": "Llama-3.1-70B", "params_billion": 70, "d_model": 8192, "n_heads": 64, "n_layers": 80, "max_context_window": 131072, "d_head": 128},
        {"name": "Mistral-7B-v0.3", "params_billion": 7, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 32768, "d_head": 128}
        # Add or comment out model specifications as needed
    ]

    BYTES_IN_GB = 1_073_741_824  # 1 GB = 1,073,741,824 bytes

    def calc_kv_cache_size_per_token(n_layers, d_model):
        return 2 * 2 * n_layers * d_model / BYTES_IN_GB  # GB/token

    def calc_memory_footprint(model_spec, n_concurrent_request, avg_context_window):
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"])
        target_gpu_mem = kv_cache_size_per_token * avg_context_window * n_concurrent_request + model_spec["params_billion"] * 2
        return target_gpu_mem

    print(f"\n******************** Estimate LLM Memory Footprint ********************")
    memory_footprint_table = []
    for model_spec in model_specs:
        memory_footprint = calc_memory_footprint(model_spec, n_concurrent_request, avg_context_window)
        memory_footprint_table.append([model_spec['name'], f"{memory_footprint:.2f} GB"])
    print(tabulate(memory_footprint_table, headers=['Model', 'Memory Footprint'], tablefmt='orgtbl'))

    def calc_kv_cache_tokens(num_gpu, gpu_memory_gb, model_params_billion, kv_cache_size):
        result = (num_gpu * gpu_memory_gb - 2 * model_params_billion) / kv_cache_size
        return result if result >= 0 else "NA"

    def calc_prefill_time_per_token(num_gpu, model_params_billion, fp16_tflops):
        result = (2 * model_params_billion / num_gpu) / fp16_tflops
        return result if result >= 0 else "NA"

    def calc_generation_time_per_token(num_gpu, model_params_billion, memory_bandwidth_gbps):
        result = (2 * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000
        return result if result >= 0 else "NA"

    def calc_estimated_response_time(prefill_time, generation_time, prompt_size, response_size):
        if isinstance(prefill_time, str) or isinstance(generation_time, str):  # Check if any are "NA"
            return "NA"
        return (prompt_size * prefill_time + response_size * generation_time) / 1000  # convert ms to seconds

    print(f"\n******************** Estimate LLM Capacity and Latency ******************** ")
    capacity_latency_table = []
    for model in model_specs:
        # print(f"Model: {model['name']} ({model['params_billion']}B parameters)")
        kv_cache_size = calc_kv_cache_size_per_token(model['n_layers'], model['d_model'])
        for gpu in gpu_specs:
            kv_cache_tokens = calc_kv_cache_tokens(num_gpu, gpu['memory_gb'], model['params_billion'], kv_cache_size)
            prefill_time_per_token = calc_prefill_time_per_token(num_gpu, model['params_billion'], gpu['fp16_tflops'])
            generation_time_per_token = calc_generation_time_per_token(num_gpu, model['params_billion'], gpu['memory_bandwidth_gbps'])
            estimated_response_time = calc_estimated_response_time(prefill_time_per_token, generation_time_per_token, prompt_size, response_size)
            capacity_latency_table.append([model['name'], gpu['name'], f"{kv_cache_tokens}", f"{prefill_time_per_token:.3f} ms", f"{generation_time_per_token:.3f} ms", f"{estimated_response_time:.1f} s"])
    print(tabulate(capacity_latency_table, headers=['Model', 'GPU', 'KV Cache Tokens', 'Prefill Time', 'Generation Time', 'Estimated Response Time'], tablefmt='orgtbl'))

if __name__ == '__main__':
    main()