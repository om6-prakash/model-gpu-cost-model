# LLM_Sizing_Guide
A calculator to estimate the memory footprint, capacity, and latency based on your planned LLM application's requirements on different GPU architectures..

# Usage
Prerequisite: `pip install -r requirements.txt`

Here are the Flags and their abbreviations for the script.
- num_gpu ('-g'): Specify the number of GPUs you plan to use for your deployment.
- prompt_sz ('-p'): Define the average size of the input prompts you expect to process.
- response_sz ('-r'): Set the average size of the responses you expect to generate.
- n_concurrent_req ('-c'): Indicate the number of concurrent requests you anticipate handling.

By modifying these variables, you can easily estimate the performance characteristics of your LLM deployment and make informed decisions about your infrastructure requirements.

# Sample output
```bash
✗ python LLM_size_pef_calculator.py -g 4 -p 4096 -r 256 -c 10
 num_gpu = 4, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10

******************** Estimate LLM Memory Footprint ********************
| Model         | KV Cache Size per Token   | Memory Footprint   |
|---------------+---------------------------+--------------------|
| Llama-3.1-70B | 0.002441 GiB/token        | 246.25 GB          |

******************** Estimate LLM Capacity and Latency ********************
| Model         | GPU       |   Max # KV Cache Tokens | Prefill Time   | TPOT (ms)   | TTFT    | E2E Latency   | Output Tokens Throughput   |
|---------------+-----------+-------------------------+----------------+-------------+---------+---------------+----------------------------|
| Llama-3.1-70B | H100 PCIe |                   73728 | 0.046 ms       | 17.500 ms   | 0.064 s | 4.7 s         | 54.82 tokens/sec           |
| Llama-3.1-70B | H100 SXM  |                   73728 | 0.035 ms       | 10.448 ms   | 0.046 s | 2.8 s         | 90.80 tokens/sec           |
| Llama-3.1-70B | H100 NVL  |                   96665 | 0.042 ms       | 8.974 ms    | 0.051 s | 2.5 s         | 103.68 tokens/sec          |
| Llama-3.1-70B | H200 SXM  |                  173670 | 0.035 ms       | 7.292 ms    | 0.043 s | 2.0 s         | 127.27 tokens/sec          |
| Llama-3.1-70B | H200 NVL  |                  173670 | 0.042 ms       | 7.292 ms    | 0.049 s | 2.0 s         | 125.60 tokens/sec          |
```
