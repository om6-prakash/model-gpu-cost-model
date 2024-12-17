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
✗ python LLM_size_pef_calculator.py -g 1 -p 4096 -r 256 -c 10
 num_gpu = 1, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10

!!!! Warning Llama-3-8B: OOM for your input=4096 and output=256 for 1x A10
Max number of concurrent requests that can be set for this use case: 3, ignore the rows in the second table which contains A10

!!!! Warning Llama-3-8B: OOM for your input=4096 and output=256 for 1x A30
Max number of concurrent requests that can be set for this use case: 3, ignore the rows in the second table which contains A30

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x A10
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains A10

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x A30
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains A30

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x L40
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains L40

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x L40s
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains L40s

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x A100 40 GB
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains A100 40 GB

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x A100 40 GB SXM
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains A100 40 GB SXM

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x A100 80 GB PCIe
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains A100 80 GB PCIe

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x A100 80 GB SXM
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains A100 80 GB SXM

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x H100 NVL
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains H100 NVL

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x H200 SXM
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains H200 SXM

!!!! Warning Llama-3-70B: OOM for your input=4096 and output=256 for 1x H200 NVL
Max number of concurrent requests that can be set for this use case: 0, ignore the rows in the second table which contains H200 NVL

******************** Estimate LLM Memory Footprint ********************
| Model       | KV Cache Size per Token   | Memory Footprint   |
|-------------+---------------------------+--------------------|
| Llama-3-8B  | 0.000488 GiB/token        | 37.25 GB           |
| Llama-3-70B | 0.002441 GiB/token        | 246.25 GB          |

******************** Estimate LLM Capacity and Latency ********************
| Model       | GPU             |   Max # KV Cache Tokens | Prefill Time   | TPOT (ms)   | TTFT    | E2E Latency   | Output Tokens Throughput   |
|-------------+-----------------+-------------------------+----------------+-------------+---------+---------------+----------------------------|
| Llama-3-8B  | A10             |                   16384 | 0.128 ms       | 26.667 ms   | 0.155 s | 7.4 s         | 34.83 tokens/sec           |
| Llama-3-8B  | A30             |                   16384 | 0.048 ms       | 17.149 ms   | 0.066 s | 4.6 s         | 55.79 tokens/sec           |
| Llama-3-8B  | L40             |                   65536 | 0.088 ms       | 18.519 ms   | 0.107 s | 5.1 s         | 50.17 tokens/sec           |
| Llama-3-8B  | L40s            |                   65536 | 0.044 ms       | 18.519 ms   | 0.063 s | 4.9 s         | 52.01 tokens/sec           |
| Llama-3-8B  | A100 40 GB      |                   49152 | 0.051 ms       | 10.289 ms   | 0.062 s | 2.8 s         | 90.01 tokens/sec           |
| Llama-3-8B  | A100 40 GB SXM  |                   49152 | 0.051 ms       | 10.289 ms   | 0.062 s | 2.8 s         | 90.01 tokens/sec           |
| Llama-3-8B  | A100 80 GB PCIe |                  131072 | 0.051 ms       | 8.269 ms    | 0.060 s | 2.3 s         | 110.02 tokens/sec          |
| Llama-3-8B  | A100 80 GB SXM  |                  131072 | 0.051 ms       | 7.847 ms    | 0.059 s | 2.2 s         | 115.37 tokens/sec          |
| Llama-3-8B  | H100 NVL        |                  159744 | 0.010 ms       | 4.103 ms    | 0.014 s | 1.1 s         | 234.98 tokens/sec          |
| Llama-3-8B  | H200 SXM        |                  256000 | 0.008 ms       | 3.333 ms    | 0.011 s | 0.9 s         | 288.79 tokens/sec          |
| Llama-3-8B  | H200 NVL        |                  256000 | 0.010 ms       | 3.333 ms    | 0.013 s | 0.9 s         | 286.82 tokens/sec          |
| Llama-3-70B | A10             |                       0 | 1.120 ms       | 233.333 ms  | 1.353 s | 64.3 s        | 3.98 tokens/sec            |
| Llama-3-70B | A30             |                       0 | 0.424 ms       | 150.054 ms  | 0.574 s | 40.2 s        | 6.38 tokens/sec            |
| Llama-3-70B | L40             |                       0 | 0.773 ms       | 162.037 ms  | 0.936 s | 44.6 s        | 5.73 tokens/sec            |
| Llama-3-70B | L40s            |                       0 | 0.387 ms       | 162.037 ms  | 0.549 s | 43.1 s        | 5.94 tokens/sec            |
| Llama-3-70B | A100 40 GB      |                       0 | 0.449 ms       | 90.032 ms   | 0.539 s | 24.9 s        | 10.29 tokens/sec           |
| Llama-3-70B | A100 40 GB SXM  |                       0 | 0.449 ms       | 90.032 ms   | 0.539 s | 24.9 s        | 10.29 tokens/sec           |
| Llama-3-70B | A100 80 GB PCIe |                       0 | 0.449 ms       | 72.351 ms   | 0.521 s | 20.4 s        | 12.57 tokens/sec           |
| Llama-3-70B | A100 80 GB SXM  |                       0 | 0.449 ms       | 68.661 ms   | 0.517 s | 19.4 s        | 13.19 tokens/sec           |
| Llama-3-70B | H100 NVL        |                       0 | 0.084 ms       | 35.897 ms   | 0.120 s | 9.5 s         | 26.85 tokens/sec           |
| Llama-3-70B | H200 SXM        |                     409 | 0.071 ms       | 29.167 ms   | 0.100 s | 7.8 s         | 33.00 tokens/sec           |
| Llama-3-70B | H200 NVL        |                     409 | 0.084 ms       | 29.167 ms   | 0.113 s | 7.8 s         | 32.78 tokens/sec           |

```
