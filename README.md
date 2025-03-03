# LLM_Sizing_Guide
A suite of calculators to help plan and optimize your LLM deployment infrastructure:
1. LLM Size & Performance Calculator - Analyze memory footprint and performance metrics
2. LLM GPU Requirements Calculator - Determine optimal GPU configuration based on performance targets

## Installation
```bash
pip install -r requirements.txt
```

## LLM Size & Performance Calculator
Estimates memory footprint, capacity, and latency for different GPU architectures.

### Usage
```bash
python LLM_size_pef_calculator.py [-g NUM_GPU] [-p PROMPT_SZ] [-r RESPONSE_SZ] [-c N_CONCURRENT_REQ] [--precision PRECISION]
```

### Arguments
- `-g, --num_gpu`: Number of GPUs (default: 1)
- `-p, --prompt_sz`: Prompt size in tokens (default: 4096)
- `-r, --response_sz`: Response size in tokens (default: 256)
- `-c, --n_concurrent_req`: Number of concurrent requests (default: 10)
- `--precision`: Precision level ['int8', 'fp8', 'fp16', 'bf16', 'tf32', 'fp32', 'fp64'] (default: 'fp16')

### Sample Output
```bash
✗ python LLM_size_pef_calculator.py -g 4 -p 4096 -r 256 -c 10
num_gpu = 4, prompt_size = 4096 tokens, response_size = 256 tokens
n_concurrent_request = 10, precision = fp16

******************** Estimate LLM Memory Footprint ********************
| Model         | KV Cache Size per Token | Memory Footprint |
|---------------+-----------------------+-----------------|
| Llama-3.1-70B | 0.002441 GiB/token    | 246.25 GB       |

******************** Estimate LLM Performance with FP16 Precision ********************
| Model         | GPU       | Memory Status | Max KV Cache Tokens | Prefill (fp16) | TPOT (ms) | TTFT    | E2E Latency | Token Rate (tokens/sec) |
|---------------+-----------+---------------+-------------------+---------------+-----------+---------+-------------+----------------------|
| Llama-3.1-70B | H100 PCIe | Fits         | 73728            | 0.046 ms      | 17.500 ms | 0.064 s | 4.7 s       | 54.82                 |
| Llama-3.1-70B | H100 SXM  | Fits         | 73728            | 0.035 ms      | 10.448 ms | 0.046 s | 2.8 s       | 90.80                 |
```

## LLM GPU Requirements Calculator
Determines optimal GPU configuration based on your performance targets.

### Usage
```bash
python llm_gpu_calculator.py -m MODEL -t TOKEN_RATE -l MAX_LATENCY [-p PROMPT_SZ] [-r RESPONSE_SZ] [-w PRECISION] [-c MAX_CONCURRENT]
```

### Arguments
- `-m, --model`: Model name (required)
- `-t, --token_rate`: Desired token rate in tokens/sec (required)
- `-l, --max_latency`: Maximum acceptable E2E latency in seconds (required)
- `-p, --prompt_sz`: Prompt size in tokens (default: 4096)
- `-r, --response_sz`: Response size in tokens (default: 256)
- `-w, --precision`: Precision level (default: 'fp16')
- `-c, --max_concurrent`: Maximum concurrent requests (optional)

### Sample Output
```bash
✗ python llm_gpu_calculator.py -m "Llama-3.1-70B" -t 100 -l 3.0

*** GPU Requirements for Llama-3.1-70B ***
Target token rate: 100 tokens/sec
Maximum latency: 3.0 seconds
Prompt size: 4096 tokens, Response size: 256 tokens
Precision: fp16
Required concurrent requests: 12

Memory Requirements:
KV Cache per token: 0.002441 GB/token
Memory per request: 10.62 GB
Model parameters: 140.00 GB
Total memory required: 267.44 GB

GPU Requirements Analysis:
| GPU Model  | GPUs Needed | Meets Requirements | E2E Latency | Throughput    | Monthly Opex | Total Capex |
|-----------+-------------+-------------------+-------------+--------------+--------------+-------------|
| H100 SXM  | 4          | Yes               | 2.8 s       | 109.7 t/s    | $10,800      | $140,000    |
| H200 SXM  | 3          | Yes               | 2.0 s       | 153.6 t/s    | $9,000       | $120,000    |

Recommended Configuration:
- 3x H200 SXM GPUs
- Expected throughput: 153.6 tokens/sec
- Expected latency: 2.0 s
- Monthly operating cost: $9,000
- Total acquisition cost: $120,000
```

## Supported Models
- DeepSeek Series (R1-8B, R1-33B, R1-70B, V2-236B, R1-671B)
- Llama Series (3-8B, 3-70B, 3.1-405B)

## Supported GPUs
- NVIDIA A100 (40GB SXM, 80GB PCIe/SXM)
- NVIDIA H100 (PCIe, SXM, NVL)
- NVIDIA H200 (SXM, NVL)
- NVIDIA B100/B200 (PCIe, SXM)
- Grace Hopper (GH100, GH200)
- Grace Blackwell (GB100, GB200)

## Notes
- All calculations are estimates and actual performance may vary
- Cost estimates are approximate and may vary by region and provider
- Memory calculations include model parameters and KV cache requirements
- Token rates are theoretical maximums; actual rates may be lower due to various factors
