import argparse
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-g', '--num_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('-p', '--prompt_sz', type=int, default=4096, help='Prompt size in tokens')
    parser.add_argument('-r', '--response_sz', type=int, default=256, help='Response size in tokens')
    parser.add_argument('-c', '--n_concurrent_req', type=int, default=10, help='Number of concurrent requests')

    args = parser.parse_args()

    num_gpu = args.num_gpu
    prompt_size = args.prompt_sz
    response_size = args.response_sz
    n_concurrent_request = args.n_concurrent_req

    print(f" num_gpu = {num_gpu}, prompt_size = {prompt_size} tokens, response_size = {response_size} tokens")
    print(f" n_concurrent_request = {n_concurrent_request}")

    gpu_specs = [
        {"name": "A10", "fp16_tflops": 125, "memory_gb": 24, "memory_bandwidth_gbps": 600},
        {"name": "A30", "fp16_tflops": 330, "memory_gb": 24, "memory_bandwidth_gbps": 933},
        {"name": "L40", "fp16_tflops": 181, "memory_gb": 48, "memory_bandwidth_gbps": 864},
        {"name": "L40s", "fp16_tflops": 362, "memory_gb": 48, "memory_bandwidth_gbps": 864},
        {"name": "A100 40 GB", "fp16_tflops": 312, "memory_gb": 40, "memory_bandwidth_gbps": 1555},
        {"name": "A100 40 GB SXM", "fp16_tflops": 312, "memory_gb": 40, "memory_bandwidth_gbps": 1555},
        {"name": "A100 80 GB PCIe", "fp16_tflops": 312, "memory_gb": 80, "memory_bandwidth_gbps": 1935},
        {"name": "A100 80 GB SXM", "fp16_tflops": 312, "memory_gb": 80, "memory_bandwidth_gbps": 2039},
        {"name": "H100 PCIe", "fp16_tflops": 756.5, "memory_gb": 80, "memory_bandwidth_gbps": 2000},
        {"name": "H100 SXM", "fp16_tflops": 989.5, "memory_gb": 80, "memory_bandwidth_gbps": 3350},
        {"name": "H100 NVL", "fp16_tflops": 835.5, "memory_gb": 94, "memory_bandwidth_gbps": 3900},
        {"name": "H200 SXM", "fp16_tflops": 989.5, "memory_gb": 141, "memory_bandwidth_gbps": 4800},
        {"name": "H200 NVL", "fp16_tflops": 835.5, "memory_gb": 141, "memory_bandwidth_gbps": 4800},
        {"name": "MI300X", "fp16_tflops": 1307, "memory_gb": 192, "memory_bandwidth_gbps": 5300}
    ]

    model_specs = [
        {"name": "Llama-3-8B", "params_billion": 8, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 8192, "d_head": 128},
        {"name": "Llama-3-70B", "params_billion": 70, "d_model": 8192, "n_heads": 64, "n_layers": 80, "max_context_window": 8192, "d_head": 128},
        {"name": "Llama-3.1-8B", "params_billion": 8, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 131072, "d_head": 128},
        {"name": "Llama-3.1-70B", "params_billion": 70, "d_model": 8192, "n_heads": 64, "n_layers": 80, "max_context_window": 131072, "d_head": 128},
        {"name": "Mistral-7B-v0.3", "params_billion": 7, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 32768, "d_head": 128},
        {"name": "Falcon-7B", "params_billion": 7, "d_model": 4544, "n_heads": 71, "n_layers": 32, "max_context_window": 2048, "d_head": 64},
        {"name": "Falcon-40B", "params_billion": 40, "d_model": 8192, "n_heads": 128, "n_layers": 60, "max_context_window": 2048, "d_head": 64},
        {"name": "Falcon-180B", "params_billion": 180, "d_model": 14848, "n_heads": 232, "n_layers": 80, "max_context_window": 2048, "d_head": 64}
        # Add or comment out model specifications as needed
    ]

    BYTES_IN_GB = 1_073_741_824

    def calc_kv_cache_size_per_token(n_layers, d_model):
        return 2 * 2 * n_layers * d_model / BYTES_IN_GB

    def calc_memory_footprint(model_spec, n_concurrent_request, context_window):
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"])
        return kv_cache_size_per_token * context_window * n_concurrent_request + model_spec["params_billion"] * 2

    def calc_kv_cache_tokens(num_gpu, gpu_memory_gb, model_params_billion, kv_cache_size):
        result = (num_gpu * gpu_memory_gb - 2 * model_params_billion) / kv_cache_size
        return result if result >= 0 else 0

    def calc_prefill_time_per_token(num_gpu, model_params_billion, gpu_fp16_tflops):
        result = (2 * model_params_billion / num_gpu) / gpu_fp16_tflops
        return result if result >= 0 else "OOM"

    def calc_tpot(num_gpu, model_params_billion, memory_bandwidth_gbps):
        result = (2 * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000
        return result if result >= 0 else "OOM"

    def calc_e2e_latency(prefill_time_per_token, tpot, prompt_size, response_size):
        return (prompt_size * prefill_time_per_token + response_size * tpot) / 1000

    print(f"\n******************** Estimate LLM Memory Footprint ********************")
    memory_footprint_table = []
    for model_spec in model_specs:
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"])
        context_window = prompt_size + response_size
        memory_footprint = calc_memory_footprint(model_spec, n_concurrent_request, context_window)
        memory_footprint_table.append([model_spec['name'], f"{kv_cache_size_per_token:.6f} GiB/token", f"{memory_footprint:.2f} GB"])
    print(tabulate(memory_footprint_table, headers=['Model', 'KV Cache Size per Token', 'Memory Footprint'], tablefmt='orgtbl'))


    for model in model_specs:
        for gpu in gpu_specs:
            kv_cache_size_per_token = calc_kv_cache_size_per_token(model["n_layers"], model["d_model"])
            context_window = prompt_size + response_size
            memory_footprint = calc_memory_footprint(model, n_concurrent_request, context_window)

            available_memory = num_gpu * gpu["memory_gb"]
            if memory_footprint > available_memory:
                print(f"\n!!!! Warning {model['name']}: n_concurrent_request={n_concurrent_request} is TOO Large!!!\nCausing OOM with ISL={prompt_size} and OSL={response_size} using {num_gpu}x {gpu['name']}")
                kv_cache_tokens = calc_kv_cache_tokens(num_gpu, gpu["memory_gb"], model["params_billion"], kv_cache_size_per_token)
                max_n_concurrent_req = int(kv_cache_tokens // context_window)
                print(f"Max number of concurrent requests that can be set for this use case: {max_n_concurrent_req}\nIgnore the rows in the following table which contains {gpu['name']} and rerun the calculator with this number")
                # exit(1)

    print(f"\n******************** Estimate LLM Capacity and Latency ******************** ")
    capacity_latency_table = []
    for model in model_specs:
        kv_cache_size = calc_kv_cache_size_per_token(model['n_layers'], model['d_model'])
        for gpu in gpu_specs:
            kv_cache_tokens = calc_kv_cache_tokens(num_gpu, gpu['memory_gb'], model['params_billion'], kv_cache_size)
            prefill_time_per_token = calc_prefill_time_per_token(num_gpu, model['params_billion'], gpu['fp16_tflops'])
            tpot = calc_tpot(num_gpu, model['params_billion'], gpu['memory_bandwidth_gbps'])
            if isinstance(prefill_time_per_token, str) or isinstance(tpot, str):
                ttft = "OOM"
                e2e_latency = "OOM"
                throughput = "OOM"
            else:
                ttft = prefill_time_per_token + tpot / 1000
                e2e_latency = calc_e2e_latency(prefill_time_per_token, tpot, prompt_size, response_size)
                throughput = response_size / e2e_latency if e2e_latency > 0 else "OOM"

            capacity_latency_table.append([
                model['name'], gpu['name'], f"{int(kv_cache_tokens)}", f"{prefill_time_per_token:.3f} ms",
                f"{tpot:.3f} ms", f"{ttft:.3f} s", f"{e2e_latency:.1f} s", f"{throughput:.2f} tokens/sec"
            ])
    print(tabulate(capacity_latency_table, headers=['Model', 'GPU', 'Max # KV Cache Tokens', 'Prefill Time', 'TPOT (ms)', 'TTFT', 'E2E Latency', 'Output Tokens Throughput'], tablefmt='orgtbl'))

if __name__ == '__main__':
    main()
