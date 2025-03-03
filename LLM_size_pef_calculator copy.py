import argparse
import os
import csv
from tabulate import tabulate

def read_tsv_file(file_path):
    """Read data from a TSV file and return a list of dictionaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        data = []
        for row in reader:
            # Convert string values to appropriate types
            processed_row = {}
            for key, value in row.items():
                if value == "None" or value == "":
                    processed_row[key] = None
                elif key in ["name", "connectivity", "architecture"]:
                    processed_row[key] = value
                else:
                    try:
                        processed_row[key] = float(value)
                    except ValueError:
                        processed_row[key] = value
            data.append(processed_row)
    return data

def main():
    parser = argparse.ArgumentParser(description='GPU Performance Calculator for LLMs')
    parser.add_argument('-g', '--num_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('-p', '--prompt_sz', type=int, default=4096, help='Prompt size in tokens')
    parser.add_argument('-r', '--response_sz', type=int, default=256, help='Response size in tokens')
    parser.add_argument('-c', '--n_concurrent_req', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('--precision', type=str, default='fp16', 
                       choices=['int8', 'fp8', 'fp16', 'bf16', 'tf32', 'fp32', 'fp64'], 
                       help='Precision level to use for calculations')
    parser.add_argument('--gpu_file', type=str, default='data/gpu_specs.tsv', 
                       help='Path to the TSV file containing GPU specifications')
    parser.add_argument('--model_file', type=str, default='data/model_specs.tsv', 
                       help='Path to the TSV file containing model specifications')

    args = parser.parse_args()

    num_gpu = args.num_gpu
    prompt_size = args.prompt_sz
    response_size = args.response_sz
    n_concurrent_request = args.n_concurrent_req
    precision = args.precision
    gpu_file = args.gpu_file
    model_file = args.model_file

    print(f" num_gpu = {num_gpu}, prompt_size = {prompt_size} tokens, response_size = {response_size} tokens")
    print(f" n_concurrent_request = {n_concurrent_request}, precision = {precision}")
    print(f" GPU specs file: {gpu_file}, Model specs file: {model_file}")

    try:
        gpu_specs = read_tsv_file(gpu_file)
        model_specs = read_tsv_file(model_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the TSV files exist at the specified paths.")
        return
    except Exception as e:
        print(f"Error reading TSV files: {e}")
        return

    BYTES_IN_GB = 1_073_741_824

    # Define bytes per parameter for different precision types
    def get_bytes_per_parameter(precision):
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
        return 2 * bytes_per_parameter * n_layers * d_model / BYTES_IN_GB

    def calc_memory_footprint(model_spec, n_concurrent_request, context_window, bytes_per_parameter):
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"], bytes_per_parameter)
        model_size_gb = model_spec["params_billion"] * bytes_per_parameter
        return kv_cache_size_per_token * context_window * n_concurrent_request + model_size_gb

    def calc_kv_cache_tokens(num_gpu, gpu_memory_gb, model_params_billion, kv_cache_size, bytes_per_parameter):
        model_size_gb = model_params_billion * bytes_per_parameter
        result = (num_gpu * gpu_memory_gb - model_size_gb) / kv_cache_size
        return result if result >= 0 else 0

    def get_compute_perf_for_precision(gpu, precision):
        # Map precision to the corresponding GPU spec key
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
        if gpu_perf is None:
            return "Not Supported"
        result = (2 * model_params_billion / num_gpu) / gpu_perf
        return result if result >= 0 else "OOM"

    def calc_tpot(num_gpu, model_params_billion, memory_bandwidth_gbps):
        result = (2 * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000
        return result if result >= 0 else "OOM"

    def calc_e2e_latency(prefill_time_per_token, tpot, prompt_size, response_size):
        if isinstance(prefill_time_per_token, str) or isinstance(tpot, str):
            return "N/A"
        return (prompt_size * prefill_time_per_token + response_size * tpot) / 1000

    # Get bytes per parameter for the specified precision
    bytes_per_parameter = get_bytes_per_parameter(precision)
    print(f"Using {bytes_per_parameter} bytes per parameter for {precision} precision")

    print(f"\n******************** Estimate LLM Memory Footprint ********************")
    memory_footprint_table = []
    for model_spec in model_specs:
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"], bytes_per_parameter)
        context_window = prompt_size + response_size
        memory_footprint = calc_memory_footprint(model_spec, n_concurrent_request, context_window, bytes_per_parameter)
        memory_footprint_table.append([model_spec['name'], f"{kv_cache_size_per_token:.6f} GiB/token", f"{memory_footprint:.2f} GB"])
    print(tabulate(memory_footprint_table, headers=['Model', 'KV Cache Size per Token', 'Memory Footprint'], tablefmt='orgtbl'))

    # Check if any GPU+model combinations would be OOM with current settings
    print(f"\n******************** OOM Warnings ********************")
    oom_warnings = False
    for model in model_specs:
        for gpu in gpu_specs:
            kv_cache_size_per_token = calc_kv_cache_size_per_token(model["n_layers"], model["d_model"], bytes_per_parameter)
            context_window = prompt_size + response_size
            memory_footprint = calc_memory_footprint(model, n_concurrent_request, context_window, bytes_per_parameter)

            available_memory = num_gpu * gpu["memory_gb"]
            if memory_footprint > available_memory:
                oom_warnings = True
                print(f"\n!!!! Warning {model['name']}: n_concurrent_request={n_concurrent_request} is TOO Large!!!")
                print(f"Causing OOM with prompt={prompt_size} and response={response_size} using {num_gpu}x {gpu['name']}")
                kv_cache_tokens = calc_kv_cache_tokens(num_gpu, gpu["memory_gb"], model["params_billion"], kv_cache_size_per_token, bytes_per_parameter)
                max_n_concurrent_req = int(kv_cache_tokens // context_window)
                print(f"Max number of concurrent requests for this configuration: {max_n_concurrent_req}")
    
    if not oom_warnings:
        print("No OOM issues detected with current configuration.")

    print(f"\n******************** Estimate LLM Performance with {precision.upper()} Precision ********************")
    capacity_latency_table = []
    
    # Filter out GPUs that don't support the specified precision
    supported_gpus = []
    for gpu in gpu_specs:
        perf = get_compute_perf_for_precision(gpu, precision)
        if perf is not None:
            supported_gpus.append(gpu)
    
    if not supported_gpus:
        print(f"No GPUs in the database support {precision.upper()} precision.")
    else:
        for model in model_specs:
            kv_cache_size = calc_kv_cache_size_per_token(model['n_layers'], model['d_model'], bytes_per_parameter)
            context_window = prompt_size + response_size
            
            for gpu in supported_gpus:
                gpu_perf = get_compute_perf_for_precision(gpu, precision)
                kv_cache_tokens = calc_kv_cache_tokens(num_gpu, gpu['memory_gb'], model['params_billion'], kv_cache_size, bytes_per_parameter)
                
                if gpu_perf is None:
                    prefill_time_per_token = "Not Supported"
                    ttft = "Not Supported"
                    e2e_latency = "Not Supported"
                    throughput = "Not Supported"
                else:
                    prefill_time_per_token = calc_prefill_time_per_token(num_gpu, model['params_billion'], gpu_perf)
                
                tpot = calc_tpot(num_gpu, model['params_billion'], gpu['memory_bandwidth_gbps'])
                
                if isinstance(prefill_time_per_token, str) or isinstance(tpot, str):
                    ttft = "N/A"
                    e2e_latency = "N/A"
                    throughput = "N/A"
                else:
                    ttft = f"{(prefill_time_per_token + tpot / 1000):.3f} s"
                    e2e_latency_val = calc_e2e_latency(prefill_time_per_token, tpot, prompt_size, response_size)
                    
                    if isinstance(e2e_latency_val, str):
                        e2e_latency = e2e_latency_val
                        throughput = "N/A"
                    else:
                        e2e_latency = f"{e2e_latency_val:.1f} s"
                        throughput = f"{response_size / e2e_latency_val:.2f} tokens/sec"
                
                # Format the prefill time and tpot appropriately
                if isinstance(prefill_time_per_token, str):
                    prefill_time_formatted = prefill_time_per_token
                else:
                    prefill_time_formatted = f"{prefill_time_per_token:.3f} ms"
                    
                if isinstance(tpot, str):
                    tpot_formatted = tpot
                else:
                    tpot_formatted = f"{tpot:.3f} ms"
                
                # Add a column to indicate if this configuration fits in memory
                memory_status = "Fits" if kv_cache_tokens >= context_window * n_concurrent_request else "OOM"
                
                capacity_latency_table.append([
                    model['name'], gpu['name'], memory_status, f"{int(kv_cache_tokens)}", 
                    prefill_time_formatted, tpot_formatted, ttft, e2e_latency, throughput
                ])
                
        print(tabulate(capacity_latency_table, 
                      headers=['Model', 'GPU', 'Memory Status', 'Max KV Cache Tokens', 
                              f'Prefill ({precision})', 'TPOT (ms)', 'TTFT', 
                              'E2E Latency', 'Throughput'], 
                      tablefmt='orgtbl'))

def generate_sample_tsv_files():
    """Generate sample TSV files if they don't exist."""
    # Sample GPU specs
    gpu_specs = [
        {"name":"A10","memory_gb":24,"memory_bandwidth_gbps":600,"connectivity":"PCIe","int8_tops":250,"fp8_tflops":None,"fp16_tflops":125,"bf16_tflops":125,"tf32_tflops":62.5,"fp32_tflops":31.2,"fp64_tflops":1.2},
        {"name":"H100 PCIe","memory_gb":80,"memory_bandwidth_gbps":2000,"connectivity":"PCIe","int8_tops":1513,"fp8_tflops":3026,"fp16_tflops":756.5,"bf16_tflops":756.5,"tf32_tflops":378.2,"fp32_tflops":51,"fp64_tflops":26}
    ]
    
    # Sample model specs
    model_specs = [
        {"name": "Llama-3-8B", "params_billion": 8, "d_model": 4096, "n_heads": 32, "n_layers": 32, "max_context_window": 8192, "d_head": 128},
        {"name": "Llama-3-70B", "params_billion": 70, "d_model": 8192, "n_heads": 64, "n_layers": 80, "max_context_window": 8192, "d_head": 128}
    ]
    
    # Generate GPU specs TSV
    if not os.path.exists('gpu_specs.tsv'):
        with open('gpu_specs.tsv', 'w', newline='') as file:
            fieldnames = ["name", "memory_gb", "memory_bandwidth_gbps", "connectivity", "int8_tops", 
                          "fp8_tflops", "fp16_tflops", "bf16_tflops", "tf32_tflops", "fp32_tflops", "fp64_tflops", 
                          "architecture", "grace_memory_gb"]
            
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            for gpu in gpu_specs:
                # Ensure all fields are present
                for field in fieldnames:
                    if field not in gpu:
                        gpu[field] = None
                writer.writerow(gpu)
        
        print("Sample GPU specs file 'gpu_specs.tsv' generated.")
    
    # Generate model specs TSV
    if not os.path.exists('model_specs.tsv'):
        with open('model_specs.tsv', 'w', newline='') as file:
            fieldnames = ["name", "params_billion", "d_model", "n_heads", "n_layers", "max_context_window", "d_head"]
            
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            for model in model_specs:
                writer.writerow(model)
        
        print("Sample model specs file 'model_specs.tsv' generated.")

if __name__ == '__main__':
    # Check if TSV files exist, if not, generate samples
    if not os.path.exists('gpu_specs.tsv') or not os.path.exists('model_specs.tsv'):
        print("TSV files not found. Generating sample files...")
        generate_sample_tsv_files()
    
    main()