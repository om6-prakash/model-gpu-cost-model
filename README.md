# LLM_Sizing_Guide
A calculator to estimate the memory footprint, capacity, and latency based on your planned LLM application's requirements on different GPU architectures..

# Usage
Prerequisite: `pip install -r requirements.txt`

Here are the Flags and their abbreviations for the script.
- num_gpu ('-g'): Specify the number of GPUs you plan to use for your deployment.
- prompt_sz ('-p'): Define the average size of the input prompts you expect to process.
- response_sz ('-r'): Set the average size of the responses you expect to generate.
- n_concurrent_req ('-c'): Indicate the number of concurrent requests you anticipate handling.
- ctx_window ('-w', '-cw'): Specify the **_average_** context window size for your use case.
By modifying these variables, you can easily estimate the performance characteristics of your LLM deployment and make informed decisions about your infrastructure requirements.

# Sample output
```bash
✗ python LLM_size_pef_calculator.py -g 1 -p 4096 -r 256 -c 10 -w 1024
 num_gpu = 1, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10, avg_context_window = 1024 tokens

******************** Estimate LLM Memory Footprint ********************
| Model           | KV Cache Size per Token   | Memory Footprint   |
|-----------------+---------------------------+--------------------|
| Llama-3-8B      | 0.000488 GiB/token        | 21.00 GB           |
| Llama-3-70B     | 0.002441 GiB/token        | 165.00 GB          |
| Llama-3.1-8B    | 0.000488 GiB/token        | 21.00 GB           |
| Llama-3.1-70B   | 0.002441 GiB/token        | 165.00 GB          |
| Mistral-7B-v0.3 | 0.000488 GiB/token        | 19.00 GB           |

******************** Estimate LLM Capacity and Latency ********************
| Model           | GPU             | KV Cache Tokens   | Prefill Time   | Generation Time   | Estimated Response Time   |
|-----------------+-----------------+-------------------+----------------+-------------------+---------------------------|
| Llama-3-8B      | A10             | 16384.0           | 0.128 ms       | 26.667 ms         | 7.4 s                     |
| Llama-3-8B      | A30             | 16384.0           | 0.048 ms       | 17.149 ms         | 4.6 s                     |
| Llama-3-8B      | L40             | 65536.0           | 0.088 ms       | 18.519 ms         | 5.1 s                     |
| Llama-3-8B      | L40s            | 65536.0           | 0.044 ms       | 18.519 ms         | 4.9 s                     |
| Llama-3-8B      | A100 40 GB      | 49152.0           | 0.051 ms       | 10.289 ms         | 2.8 s                     |
| Llama-3-8B      | A100 40 GB SXM  | 49152.0           | 0.051 ms       | 10.289 ms         | 2.8 s                     |
| Llama-3-8B      | A100 80 GB PCIe | 131072.0          | 0.051 ms       | 8.269 ms          | 2.3 s                     |
| Llama-3-8B      | A100 80 GB SXM  | 131072.0          | 0.051 ms       | 7.847 ms          | 2.2 s                     |
| Llama-3-8B      | H100 PCIe       | 131072.0          | 0.011 ms       | 8.000 ms          | 2.1 s                     |
| Llama-3-8B      | H100 SXM        | 131072.0          | 0.008 ms       | 4.776 ms          | 1.3 s                     |
| Llama-3-8B      | H100 NVL        | 352256.0          | 0.004 ms       | 2.051 ms          | 0.5 s                     |
| Llama-3-70B     | A10             | NA                | 1.120 ms       | 233.333 ms        | 64.3 s                    |
| Llama-3-70B     | A30             | NA                | 0.424 ms       | 150.054 ms        | 40.2 s                    |
| Llama-3-70B     | L40             | NA                | 0.773 ms       | 162.037 ms        | 44.6 s                    |
| Llama-3-70B     | L40s            | NA                | 0.387 ms       | 162.037 ms        | 43.1 s                    |
| Llama-3-70B     | A100 40 GB      | NA                | 0.449 ms       | 90.032 ms         | 24.9 s                    |
| Llama-3-70B     | A100 40 GB SXM  | NA                | 0.449 ms       | 90.032 ms         | 24.9 s                    |
| Llama-3-70B     | A100 80 GB PCIe | NA                | 0.449 ms       | 72.351 ms         | 20.4 s                    |
| Llama-3-70B     | A100 80 GB SXM  | NA                | 0.449 ms       | 68.661 ms         | 19.4 s                    |
| Llama-3-70B     | H100 PCIe       | NA                | 0.093 ms       | 70.000 ms         | 18.3 s                    |
| Llama-3-70B     | H100 SXM        | NA                | 0.071 ms       | 41.791 ms         | 11.0 s                    |
| Llama-3-70B     | H100 NVL        | 19660.8           | 0.035 ms       | 17.949 ms         | 4.7 s                     |
```
