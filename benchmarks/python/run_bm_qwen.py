import subprocess
import os

model = "qwen_7b_chat"
mode = "plugin"
batch_sizes = "1"
log_level = "info"
engine_dir = "../../examples/qwenvl/trt_engines/Qwen-7B-Chat-int4-gptq"
dir_name = "bench_result/qwen_7b-int4-gptq"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

enable_cuda_graph = True

input_output_lens = [
    "128,128",
    "512,512",
    "1024,512",
]


for input_output_len in input_output_lens:
    command = f"python3 benchmark.py \
        --model {model} \
        --mode {mode} \
        --batch_size \"{batch_sizes}\" \
        --input_output_len \"{input_output_len}\" \
        --log_level \"{log_level}\" \
        --engine_dir {engine_dir} \
        {'--enable_cuda_graph' if enable_cuda_graph else ''}"
    log_file = dir_name.split('/')[-1] + f"_{input_output_len.replace(',', '_')}.log"
    print(command)
    log_file_dir = os.path.join(dir_name,log_file)
    subprocess.call(f"{command} > {log_file_dir} 2>&1", shell=True)
    command2 = f"sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches"
    subprocess.call(f"{command2} ", shell=True)
