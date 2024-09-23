# LLM_Sizing_Guide
A calculator to estimate the memory footprint, capacity, and latency based on your planned LLM application's requirements or your existing infrastructure.

# Usage
Here are the Flags and their abbreviations for the script
- num_gpu ('-g'): Specify the number of GPUs you plan to use for your deployment.
- prompt_sz ('-p'): Define the average size of the input prompts you expect to process.
- response_sz ('-r'): Set the average size of the responses you expect to generate.
- n_concurrent_req ('-c'): Indicate the number of concurrent requests you anticipate handling.
- ctx_window ('-w', '-cw'): Specify the **_average_** context window size for your use case.
By modifying these variables, you can easily estimate the performance characteristics of your LLM deployment and make informed decisions about your infrastructure requirements.

# Sample output
```bash
$ python LLM_size_pef_calculator.py -g 1 -p 4096 -r 256 -c 10 -w 1024
 num_gpu = 1, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10, avg_context_window = 1024 tokens

******************** Estimate LLM Memory Footprint ********************
Model: Llama-3-8B, kv_cache_size_per_token: 0.000488 GiB/token, Memory Footprint: 21.00 GB
Model: Llama-3-70B, kv_cache_size_per_token: 0.002441 GiB/token, Memory Footprint: 165.00 GB
Model: Llama-3.1-8B, kv_cache_size_per_token: 0.000488 GiB/token, Memory Footprint: 21.00 GB
Model: Llama-3.1-70B, kv_cache_size_per_token: 0.002441 GiB/token, Memory Footprint: 165.00 GB
Model: Mistral-7B-v0.3, kv_cache_size_per_token: 0.000488 GiB/token, Memory Footprint: 19.00 GB

******************** Estimate LLM Capacity and Latency ********************
Model: Llama-3-8B (8B parameters)
  GPU: A10, KV Cache Tokens: 16384.0, Prefill Time: 0.128 ms, Generation Time: 26.667 ms, Estimated Response Time: 7.4 s
  GPU: A30, KV Cache Tokens: 16384.0, Prefill Time: 0.048 ms, Generation Time: 17.149 ms, Estimated Response Time: 4.6 s
  GPU: L40, KV Cache Tokens: 65536.0, Prefill Time: 0.088 ms, Generation Time: 18.519 ms, Estimated Response Time: 5.1 s
  GPU: L40s, KV Cache Tokens: 65536.0, Prefill Time: 0.044 ms, Generation Time: 18.519 ms, Estimated Response Time: 4.9 s
  GPU: A100 40 GB, KV Cache Tokens: 49152.0, Prefill Time: 0.051 ms, Generation Time: 10.289 ms, Estimated Response Time: 2.8 s
  GPU: A100 40 GB SXM, KV Cache Tokens: 49152.0, Prefill Time: 0.051 ms, Generation Time: 10.289 ms, Estimated Response Time: 2.8 s
  GPU: A100 80 GB PCIe, KV Cache Tokens: 131072.0, Prefill Time: 0.051 ms, Generation Time: 8.269 ms, Estimated Response Time: 2.3 s
  GPU: A100 80 GB SXM, KV Cache Tokens: 131072.0, Prefill Time: 0.051 ms, Generation Time: 7.847 ms, Estimated Response Time: 2.2 s
  GPU: H100 PCIe, KV Cache Tokens: 131072.0, Prefill Time: 0.011 ms, Generation Time: 8.000 ms, Estimated Response Time: 2.1 s
  GPU: H100 SXM, KV Cache Tokens: 131072.0, Prefill Time: 0.008 ms, Generation Time: 4.776 ms, Estimated Response Time: 1.3 s
  GPU: H100 NVL, KV Cache Tokens: 352256.0, Prefill Time: 0.004 ms, Generation Time: 2.051 ms, Estimated Response Time: 0.5 s
Model: Llama-3-70B (70B parameters)
  GPU: A10, KV Cache Tokens: NA, Prefill Time: 1.120 ms, Generation Time: 233.333 ms, Estimated Response Time: 64.3 s
  GPU: A30, KV Cache Tokens: NA, Prefill Time: 0.424 ms, Generation Time: 150.054 ms, Estimated Response Time: 40.2 s
  GPU: L40, KV Cache Tokens: NA, Prefill Time: 0.773 ms, Generation Time: 162.037 ms, Estimated Response Time: 44.6 s
  GPU: L40s, KV Cache Tokens: NA, Prefill Time: 0.387 ms, Generation Time: 162.037 ms, Estimated Response Time: 43.1 s
  GPU: A100 40 GB, KV Cache Tokens: NA, Prefill Time: 0.449 ms, Generation Time: 90.032 ms, Estimated Response Time: 24.9 s
  GPU: A100 40 GB SXM, KV Cache Tokens: NA, Prefill Time: 0.449 ms, Generation Time: 90.032 ms, Estimated Response Time: 24.9 s
  GPU: A100 80 GB PCIe, KV Cache Tokens: NA, Prefill Time: 0.449 ms, Generation Time: 72.351 ms, Estimated Response Time: 20.4 s
  GPU: A100 80 GB SXM, KV Cache Tokens: NA, Prefill Time: 0.449 ms, Generation Time: 68.661 ms, Estimated Response Time: 19.4 s
  GPU: H100 PCIe, KV Cache Tokens: NA, Prefill Time: 0.093 ms, Generation Time: 70.000 ms, Estimated Response Time: 18.3 s
  GPU: H100 SXM, KV Cache Tokens: NA, Prefill Time: 0.071 ms, Generation Time: 41.791 ms, Estimated Response Time: 11.0 s
  GPU: H100 NVL, KV Cache Tokens: 19660.8, Prefill Time: 0.035 ms, Generation Time: 17.949 ms, Estimated Response Time: 4.7 s
```
