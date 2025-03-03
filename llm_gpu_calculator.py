import argparse
from tabulate import tabulate
import math

def main():
    parser = argparse.ArgumentParser(description='Calculate GPU Requirements for LLM Performance Targets')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    parser.add_argument('-t', '--token_rate', type=float, required=True, help='Desired token rate (tokens/sec)')
    parser.add_argument('-l', '--max_latency', type=float, required=True, help='Maximum acceptable E2E latency (seconds)')
    parser.add_argument('-p', '--prompt_sz', type=int, default=4096, help='Prompt size in tokens')
    parser.add_argument('-r', '--response_sz', type=int, default=256, help='Response size in tokens')
    parser.add_argument('-w', '--precision', type=str, default='fp16', 
                        choices=['int8', 'fp8', 'fp16', 'bf16', 'tf32', 'fp32', 'fp64'],
                        help='Precision level to use for calculations')
    parser.add_argument('-c', '--max_concurrent', type=int, default=None, 
                        help='Maximum concurrent requests (calculated from token rate if not specified)')

    args = parser.parse_args()

    model_name = args.model
    token_rate = args.token_rate
    max_latency = args.max_latency
    prompt_size = args.prompt_sz
    response_size = args.response_sz
    precision = args.precision
    max_concurrent = args.max_concurrent

    # Load GPU and model specifications
    gpu_specs = load_gpu_specs()
    model_specs = load_model_specs()

    # Find the specified model in our database
    model_spec = None
    for model in model_specs:
        if model["name"].lower() == model_name.lower():
            model_spec = model
            break
    
    if model_spec is None:
        print(f"Error: Model '{model_name}' not found in database.")
        print("Available models:")
        for model in model_specs:
            print(f"- {model['name']}")
        return

    print(f"\n*** GPU Requirements for {model_name} ***")
    print(f"Target token rate: {token_rate} tokens/sec")
    print(f"Maximum latency: {max_latency} seconds")
    print(f"Prompt size: {prompt_size} tokens, Response size: {response_size} tokens")
    print(f"Precision: {precision}")

    # Get bytes per parameter for the specified precision
    bytes_per_parameter = get_bytes_per_parameter(precision)
    
    # Calculate KV cache size per token for this model
    kv_cache_size_per_token = calc_kv_cache_size_per_token(
        model_spec["n_layers"], 
        model_spec["d_model"], 
        bytes_per_parameter
    )
    
    # If max_concurrent is not specified, calculate it from the token rate
    context_window = prompt_size + response_size
    if max_concurrent is None:
        # Estimate concurrent requests needed to achieve target token rate
        # Each request generates response_size tokens in max_latency seconds
        max_concurrent = math.ceil(token_rate * max_latency / response_size)
    
    print(f"Required concurrent requests: {max_concurrent}")
    
    # Calculate memory required per request
    memory_per_request = kv_cache_size_per_token * context_window
    
    # Calculate total memory required for KV cache
    total_kv_memory = memory_per_request * max_concurrent
    
    # Calculate memory required for model parameters
    model_memory = model_spec["params_billion"] * bytes_per_parameter
    
    # Calculate total memory required
    total_memory_required = total_kv_memory + model_memory
    
    print(f"\nMemory Requirements:")
    print(f"KV Cache per token: {kv_cache_size_per_token:.6f} GB/token")
    print(f"Memory per request: {memory_per_request:.2f} GB")
    print(f"Model parameters: {model_memory:.2f} GB")
    print(f"Total memory required: {total_memory_required:.2f} GB")

    results = []
    
    # For each GPU type, calculate how many would be needed
    for gpu in gpu_specs:
        # Skip GPUs that don't support the specified precision
        gpu_perf = get_compute_perf_for_precision(gpu, precision)
        if gpu_perf is None:
            continue
        
        # Calculate prefill time per token
        min_gpus_for_compute = 1
        prefill_time = calc_prefill_time_per_token(min_gpus_for_compute, model_spec["params_billion"], gpu_perf)
        
        while isinstance(prefill_time, str) or prefill_time * prompt_size / 1000 > max_latency:
            min_gpus_for_compute += 1
            prefill_time = calc_prefill_time_per_token(min_gpus_for_compute, model_spec["params_billion"], gpu_perf)
            if min_gpus_for_compute > 128:  # Practical limit
                break
        
        # Calculate time per output token
        tpot = calc_tpot(min_gpus_for_compute, model_spec["params_billion"], gpu["memory_bandwidth_gbps"])
        
        # Calculate e2e latency
        if isinstance(prefill_time, str) or isinstance(tpot, str):
            e2e_latency = "N/A"
            gpus_needed = "N/A"
            continue
        
        e2e_latency = calc_e2e_latency(prefill_time, tpot, prompt_size, response_size)
        
        # Check if this configuration meets latency requirements
        meets_latency = e2e_latency <= max_latency
        
        # Calculate GPUs needed for memory
        gpus_for_memory = math.ceil(total_memory_required / gpu["memory_gb"])
        
        # The total GPUs needed is the max of compute and memory requirements
        gpus_needed = max(min_gpus_for_compute, gpus_for_memory)
        
        # Calculate the actual throughput with this many GPUs
        # Adjust prefill and tpot for the actual number of GPUs
        actual_prefill = calc_prefill_time_per_token(gpus_needed, model_spec["params_billion"], gpu_perf)
        actual_tpot = calc_tpot(gpus_needed, model_spec["params_billion"], gpu["memory_bandwidth_gbps"])
        actual_e2e = calc_e2e_latency(actual_prefill, actual_tpot, prompt_size, response_size)
        
        # The throughput is the number of tokens generated per second
        if isinstance(actual_e2e, str):
            throughput = "N/A"
            meets_requirements = False
        else:
            throughput = response_size / actual_e2e * max_concurrent
            meets_requirements = actual_e2e <= max_latency and throughput >= token_rate
        
        # Time to first token
        ttft = (actual_prefill * prompt_size / 1000) + (actual_tpot / 1000)
        
        # Calculate cost metrics
        if isinstance(gpus_needed, int):
            monthly_opex = gpus_needed * gpu["opex_per_day"] * 30  # 30 days per month
            total_capex = gpus_needed * gpu["capex"]
        else:
            monthly_opex = "N/A"
            total_capex = "N/A"
        
        results.append([
            gpu["name"],
            gpus_needed,
            f"{memory_per_request * max_concurrent:.2f} GB",
            f"{gpus_for_memory} GPUs",
            f"{min_gpus_for_compute} GPUs",
            "Yes" if meets_requirements else "No",
            f"{actual_prefill:.3f} ms" if not isinstance(actual_prefill, str) else actual_prefill,
            f"{actual_tpot:.3f} ms" if not isinstance(actual_tpot, str) else actual_tpot,
            f"{ttft:.3f} s" if not isinstance(ttft, str) else ttft,
            f"{actual_e2e:.3f} s" if not isinstance(actual_e2e, str) else actual_e2e,
            f"{throughput:.2f} tokens/s" if not isinstance(throughput, str) else throughput,
            f"${monthly_opex:,.2f}" if not isinstance(monthly_opex, str) else monthly_opex,
            f"${total_capex:,.2f}" if not isinstance(total_capex, str) else total_capex
        ])
    
    # Sort results by GPUs needed (ascending)
    results.sort(key=lambda x: x[1] if isinstance(x[1], int) else float('inf'))
    
    print("\nGPU Requirements Analysis:")
    print(tabulate(results, headers=[
        'GPU Model', 
        'GPUs Needed', 
        'KV Cache Memory',
        'Memory Limited By',
        'Compute Limited By',
        'Meets Requirements',
        f'Prefill ({precision})', 
        'TPOT', 
        'TTFT', 
        'E2E Latency',
        'Throughput',
        'Monthly Opex',
        'Total Capex'
    ], tablefmt='orgtbl'))
    
    # Print recommendation
    viable_options = [r for r in results if r[5] == "Yes"]
    if viable_options:
        print("\nRecommended Configuration:")
        best_option = viable_options[0]
        print(f"- {best_option[1]}x {best_option[0]} GPUs")
        print(f"- Expected throughput: {best_option[10]}")
        print(f"- Expected latency: {best_option[9]}")
        print(f"- Monthly operating cost: {best_option[11]}")
        print(f"- Total acquisition cost: {best_option[12]}")
    else:
        print("\nNo viable configurations found that meet both token rate and latency requirements.")
        print("Consider:")
        print("- Reducing the target token rate")
        print("- Increasing the maximum acceptable latency")
        print("- Using more powerful GPUs or a different precision")

def load_gpu_specs():
    """Load GPU specifications with cost information."""
    return [
        {"name":"A100 40 GB SXM","memory_gb":40,"memory_bandwidth_gbps":1555,"connectivity":"SXM","int8_tops":624,"fp8_tflops":None,"fp16_tflops":312,"bf16_tflops":312,"tf32_tflops":156,"fp32_tflops":19.5,"fp64_tflops":9.7,"opex_per_day":40,"capex":15000},
        {"name":"A100 80 GB PCIe","memory_gb":80,"memory_bandwidth_gbps":1935,"connectivity":"PCIe","int8_tops":624,"fp8_tflops":None,"fp16_tflops":312,"bf16_tflops":312,"tf32_tflops":156,"fp32_tflops":19.5,"fp64_tflops":9.7,"opex_per_day":55,"capex":20000},
        {"name":"A100 80 GB SXM","memory_gb":80,"memory_bandwidth_gbps":2039,"connectivity":"SXM","int8_tops":624,"fp8_tflops":None,"fp16_tflops":312,"bf16_tflops":312,"tf32_tflops":156,"fp32_tflops":19.5,"fp64_tflops":9.7,"opex_per_day":60,"capex":22000},
        {"name":"H100 PCIe","memory_gb":80,"memory_bandwidth_gbps":2000,"connectivity":"PCIe","int8_tops":1513,"fp8_tflops":3026,"fp16_tflops":756.5,"bf16_tflops":756.5,"tf32_tflops":378.2,"fp32_tflops":51,"fp64_tflops":26,"opex_per_day":80,"capex":33000},
        {"name":"H100 SXM","memory_gb":80,"memory_bandwidth_gbps":3350,"connectivity":"SXM","int8_tops":1979,"fp8_tflops":3958,"fp16_tflops":989.5,"bf16_tflops":989.5,"tf32_tflops":494.7,"fp32_tflops":67,"fp64_tflops":33.5,"opex_per_day":90,"capex":35000},
        {"name":"H100 NVL","memory_gb":94,"memory_bandwidth_gbps":3900,"connectivity":"NVL","int8_tops":1671,"fp8_tflops":3342,"fp16_tflops":835.5,"bf16_tflops":835.5,"tf32_tflops":417.7,"fp32_tflops":56.5,"fp64_tflops":28.2,"opex_per_day":95,"capex":37000},
        {"name":"H200 SXM","memory_gb":141,"memory_bandwidth_gbps":4800,"connectivity":"SXM","int8_tops":1979,"fp8_tflops":3958,"fp16_tflops":989.5,"bf16_tflops":989.5,"tf32_tflops":494.7,"fp32_tflops":67,"fp64_tflops":33.5,"opex_per_day":100,"capex":40000},
        {"name":"H200 NVL","memory_gb":141,"memory_bandwidth_gbps":4800,"connectivity":"NVL","int8_tops":1671,"fp8_tflops":3342,"fp16_tflops":835.5,"bf16_tflops":835.5,"tf32_tflops":417.7,"fp32_tflops":56.5,"fp64_tflops":28.2,"opex_per_day":105,"capex":42000},
        {"name":"B100 PCIe","memory_gb":96,"memory_bandwidth_gbps":3078,"connectivity":"PCIe","int8_tops":2220,"fp8_tflops":4440,"fp16_tflops":1110,"bf16_tflops":1110,"tf32_tflops":555,"fp32_tflops":74,"fp64_tflops":37,"opex_per_day":110,"capex":45000},
        {"name":"B100 SXM","memory_gb":96,"memory_bandwidth_gbps":3078,"connectivity":"SXM","int8_tops":2664,"fp8_tflops":5328,"fp16_tflops":1332,"bf16_tflops":1332,"tf32_tflops":666,"fp32_tflops":89,"fp64_tflops":44.5,"opex_per_day":120,"capex":48000},
        {"name":"B200 PCIe","memory_gb":192,"memory_bandwidth_gbps":5376,"connectivity":"PCIe","int8_tops":2940,"fp8_tflops":5880,"fp16_tflops":1470,"bf16_tflops":1470,"tf32_tflops":735,"fp32_tflops":98,"fp64_tflops":49,"opex_per_day":140,"capex":65000},
        {"name":"B200 SXM","memory_gb":192,"memory_bandwidth_gbps":5376,"connectivity":"SXM","int8_tops":3540,"fp8_tflops":7080,"fp16_tflops":1770,"bf16_tflops":1770,"tf32_tflops":885,"fp32_tflops":118,"fp64_tflops":59,"opex_per_day":150,"capex":70000},
        {"name":"GH100 (Grace Hopper)","memory_gb":80,"memory_bandwidth_gbps":3350,"connectivity":"SXM","architecture":"Grace Hopper","grace_memory_gb":480,"int8_tops":1979,"fp8_tflops":3958,"fp16_tflops":989.5,"bf16_tflops":989.5,"tf32_tflops":494.7,"fp32_tflops":67,"fp64_tflops":33.5,"opex_per_day":110,"capex":50000},
        {"name":"GH200 (Grace Hopper)","memory_gb":141,"memory_bandwidth_gbps":4800,"connectivity":"NVL","architecture":"Grace Hopper","grace_memory_gb":480,"int8_tops":1979,"fp8_tflops":3958,"fp16_tflops":989.5,"bf16_tflops":989.5,"tf32_tflops":494.7,"fp32_tflops":67,"fp64_tflops":33.5,"opex_per_day":130,"capex":60000},
        {"name":"GB100 (Grace Blackwell)","memory_gb":96,"memory_bandwidth_gbps":3078,"connectivity":"SXM","architecture":"Grace Blackwell","grace_memory_gb":480,"int8_tops":2664,"fp8_tflops":5328,"fp16_tflops":1332,"bf16_tflops":1332,"tf32_tflops":666,"fp32_tflops":89,"fp64_tflops":44.5,"opex_per_day":140,"capex":65000},
        {"name":"GB200 (Grace Blackwell)","memory_gb":192,"memory_bandwidth_gbps":5376,"connectivity":"NVL","architecture":"Grace Blackwell","grace_memory_gb":576,"int8_tops":3540,"fp8_tflops":7080,"fp16_tflops":1770,"bf16_tflops":1770,"tf32_tflops":885,"fp32_tflops":118,"fp64_tflops":59,"opex_per_day":160,"capex":80000},
    ]

def load_model_specs():
    """Load model specifications."""
    return [
        {"name": "DeepSeek-R1-8B", "params_billion": 8, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 32768, "d_head": 128},
        {"name": "DeepSeek-R1-33B", "params_billion": 33, "d_model": 6144, "n_heads": 48, "n_layers": 48, "max_context_window": 32768, "d_head": 128},
        {"name": "DeepSeek-R1-70B", "params_billion": 70, "d_model": 8192, "n_heads": 64, "n_layers": 72, "max_context_window": 32768, "d_head": 128},
        {"name": "DeepSeek-V2-236B", "params_billion": 236, "d_model": 12288, "n_heads": 96, "n_layers": 120, "max_context_window": 32768, "d_head": 128},
        {"name": "DeepSeek-R1-671B", "params_billion": 671, "d_model": 16384, "n_heads": 128, "n_layers": 168, "max_context_window": 32768, "d_head": 128},
        # Add other models as needed
        {"name": "Llama-3-8B", "params_billion": 8, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 8192, "d_head": 128},
        {"name": "Llama-3-70B", "params_billion": 70, "d_model": 8192, "n_heads": 64, "n_layers": 80, "max_context_window": 8192, "d_head": 128},
        {"name": "Llama-3.1-405B", "params_billion": 405, "d_model": 16384, "n_heads": 128, "n_layers": 120, "max_context_window": 131072, "d_head": 128},
    ]

def get_bytes_per_parameter(precision):
    """Define bytes per parameter for different precision types."""
    precision_bytes = {
        'int8': 1,    # 1 byte for INT8
        'fp8': 1,     # 1 byte for FP8
        'fp16': 2,    # 2 bytes for FP16
        'bf16': 2,    # 2 bytes for BF16
        'tf32': 4,    # 4 bytes for TF32
        'fp32': 4,    # 4 bytes for FP32
        'fp64': 8     # 8 bytes for FP64
    }
    return precision_bytes.get(precision, 2)  # Default to 2 bytes if precision not recognized

def calc_kv_cache_size_per_token(n_layers, d_model, bytes_per_parameter):
    """Calculate KV cache size per token in GB."""
    BYTES_IN_GB = 1_073_741_824
    return 2 * bytes_per_parameter * n_layers * d_model / BYTES_IN_GB

def get_compute_perf_for_precision(gpu, precision):
    """Get the compute performance for the specified precision."""
    precision_map = {
        'int8': 'int8_tops',
        'fp8': 'fp8_tflops',
        'fp16': 'fp16_tflops',
        'bf16': 'bf16_tflops',
        'tf32': 'tf32_tflops',
        'fp32': 'fp32_tflops',
        'fp64': 'fp64_tflops'
    }
    
    key = precision_map.get(precision)
    if key is None:
        return None
    
    return gpu.get(key)

def calc_prefill_time_per_token(num_gpu, model_params_billion, gpu_perf):
    """Calculate prefill time per token in milliseconds."""
    if gpu_perf is None:
        return "Not Supported"
    result = (2 * model_params_billion / num_gpu) / gpu_perf
    return result if result >= 0 else "OOM"

def calc_tpot(num_gpu, model_params_billion, memory_bandwidth_gbps):
    """Calculate time per output token (TPOT) in milliseconds."""
    result = (2 * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000
    return result if result >= 0 else "OOM"

def calc_e2e_latency(prefill_time_per_token, tpot, prompt_size, response_size):
    """Calculate end-to-end latency in seconds."""
    if isinstance(prefill_time_per_token, str) or isinstance(tpot, str):
        return "N/A"
    return (prompt_size * prefill_time_per_token + response_size * tpot) / 1000

if __name__ == '__main__':
    main()