# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
import tensorrt_llm.profiler as profiler

from utils import (load_tokenizer, read_model_name, throttle_generator)


def get_quant_mode(quantization):
    quant_mode = QuantMode(0)
    use_smooth_quant = False
    per_token = False
    per_channel = False
    weight_only_precision = 'int8'

    if quantization == "fp8":
        quant_mode = quant_mode.set_fp8_qdq()
        quant_mode = quant_mode.set_fp8_kv_cache()

    elif quantization == "fp8_gemm":
        quant_mode = quant_mode.set_fp8_qdq()

    elif quantization == "fp8_kv_cache":
        quant_mode = quant_mode.set_fp8_kv_cache()

    elif quantization == "int8_sq_per_tensor":
        use_smooth_quant = True
        quant_mode = QuantMode.use_smooth_quant(per_token, per_channel)

    elif quantization == "int8_sq_per_token_channel":
        use_smooth_quant = True
        per_token = True
        per_channel = True
        quant_mode = QuantMode.use_smooth_quant(per_token, per_channel)

    elif quantization == "int8_weight_only":
        use_smooth_quant = False
        weight_only_precision = 'int8'
        quant_mode = QuantMode.use_weight_only(use_int4_weights=False)

    elif quantization == "int4_weight_only":
        weight_only_precision = 'int4'
        quant_mode = QuantMode.use_weight_only(use_int4_weights=True)

    elif quantization == "int4_weight_only_awq":
        weight_only_precision = 'int4_awq'
        quant_mode = QuantMode.from_description(quantize_weights=True,
                                                quantize_activations=False,
                                                per_token=False,
                                                per_channel=False,
                                                per_group=True,
                                                use_int4_weights=True)

    elif quantization == "int4_weight_only_gptq":
        weight_only_precision = 'int4_gptq'
        quant_mode = QuantMode.from_description(quantize_weights=True,
                                                quantize_activations=False,
                                                per_token=False,
                                                per_channel=False,
                                                per_group=True,
                                                use_int4_weights=True)

    elif quantization is None:
        pass

    else:
        raise Exception(f'Unexpected quantization: {quantization}')

    return quant_mode, use_smooth_quant, weight_only_precision


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        i +=1
        if not i % stream_interval:
            yield out
    
    if i % stream_interval:
        yield out


def throttle_generator_skip(generator, stream_interval, skip_first=0):
    tmp = 0
    for i, out in enumerate(generator):
        i += 1
        if i <= skip_first:
            continue
        tmp +=1 
        if tmp % stream_interval == 0 or i == skip_first + 1:
            print("========gqq i is and tmp is ",i,tmp)
            print("========gqq out is ",out)
            yield i , out


def build_inputs(tokenizer, query):
    inputs = tokenizer([query], return_tensors="pt")
    inputs = inputs.to("cuda")
    return inputs


def flatten_list(lst):
    flat_list = []
    for element in lst:
        if type(element) == list:
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list


def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_kv_heads = config['pretrained_config']['num_key_value_heads']
    num_heads = config['pretrained_config']['num_attention_heads']
    vocab_size= config['pretrained_config']['vocab_size']
    num_layers = config['pretrained_config']['num_hidden_layers']
    hidden_size = config['pretrained_config']['hidden_size']

    use_gpt_attention_plugin = config['build_config']['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['build_config']['plugin_config']['remove_input_padding']
    dtype = config['pretrained_config']['dtype']
    tp_size = 1 # config['builder_config']['tensor_parallel']
    pp_size = 1 # config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
   
    paged_kv_cache = config['build_config']['plugin_config']['paged_kv_cache']
    tokens_per_block = config['build_config']['plugin_config']['tokens_per_block']
    quant_mode, _, _ = get_quant_mode('int4_weight_only_gptq')
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    use_custom_all_reduce = config['build_config']['plugin_config'].get('use_custom_all_reduce',
                                                        False)
    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               dtype=dtype,
                               quant_mode=quant_mode,
                               use_custom_all_reduce=use_custom_all_reduce,
                               max_batch_size=1)

    return model_config, tp_size, pp_size, dtype


def parse_input(input_text: str, input_file: str, tokenizer, end_id: int,
                remove_input_padding: bool):
    input_tokens = []
    if input_file is None:
        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_tokens.append(np.array(line, dtype='int32'))
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                row = row[row != end_id]
                input_tokens.append(row)
        else:
            print('Input file format not supported.')
            raise SystemExit

    input_ids = None
    input_lengths = torch.tensor([len(x) for x in input_tokens],
                                 dtype=torch.int32,
                                 device='cuda')
    if remove_input_padding:
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                 device='cuda').unsqueeze(0)
    else:
        input_ids = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
            end_id).cuda()

    return input_ids, input_lengths


def print_output(output_ids, input_lengths, max_output_len, tokenizer,
                 output_csv, output_npy):
    num_beams = output_ids.size(1)
    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = output_ids[b][0][:input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs, skip_special_tokens=True)
            print(f'Input: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                print(f'Output: \"{output_text}\"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='llama_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument(
        '--input_tokens',
        dest='input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'llama_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    input_file: str = None,
    output_csv: str = None,
    output_npy: str = None,
    tokenizer_dir: str = None,
    num_beams: int = 1,
    streaming: bool = False,
    streaming_interval: int = 5,
):
    tensorrt_llm.logger.set_level(log_level)

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    model_config, tp_size, pp_size, dtype = read_config(config_path)
    world_size = tp_size * pp_size

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    model_name, model_version = read_model_name(args.engine_dir)
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        model_name=model_name,
        model_version=model_version,
    )
    sampling_config = SamplingConfig(end_id=end_id,
                                     pad_id=pad_id,
                                     num_beams=num_beams)
    serialize_path = engine_dir / 'rank0.engine'
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=False,
                                                     debug_tensors_to_save=None)
    if runtime_rank == 0:
        print(f"Running the {dtype} engine ...")
    run_i = 0
    while True:
        input_text = input("QUESITION (or 'q' to quit): ")        
        if input_text.lower() == 'q':
            break
        print('\n')
        print("ASSISITANT:",end='\n')
        time_stamp = 'tensorrt_llm'+str(run_i)
        profiler.start(time_stamp)
        
        input_ids = build_inputs(tokenizer, input_text)['input_ids'].int().contiguous().cuda()
        input_lengths = torch.tensor(
        [input_ids.size(1) for _ in range(input_ids.size(0))]).int().cuda()

        max_input_length = torch.max(input_lengths).item()
        decoder.setup(input_lengths.size(0), max_input_length, max_output_len,
                    num_beams)

        output_gen_ids = decoder.decode(input_ids,
                                        input_lengths,
                                        sampling_config,
                                        streaming=streaming)
        torch.cuda.synchronize()
        total_tok = 0
        if streaming:
            for idx, output_ids in enumerate((throttle_generator(output_gen_ids,
                                                streaming_interval))):
                stream_interval = streaming_interval
                ret = False
                output_ids_tmp = output_ids[:, 0, :].tolist()[0]
                output_ids_target = output_ids_tmp[idx*stream_interval+input_lengths:(idx+1)*stream_interval+input_lengths] #just 1 beam on my example
                output_text = tokenizer.decode(output_ids_target)
                total_tok += stream_interval
                if '</s>' in output_text:
                    output_text = output_text.replace('</s>', '')
                    ret  = True
                print(f"{output_text}",end='',flush=True)
                if ret:
                    break
            profiler.stop(time_stamp)
            latency = profiler.elapsed_time_in_sec(time_stamp)
            print("\n")
            print("===================================================")
            print("Speed of Inference: {:.2f} tok/s.".format(float(total_tok)/latency))
            print("===================================================")
            run_i +=1
            print("\n")
        else:
            output_ids = output_gen_ids
            if runtime_rank == 0:
                print_output(output_ids, input_lengths, max_output_len, tokenizer,
                            output_csv, output_npy)
        

if __name__ == '__main__':
    args = parse_arguments() 
    generate(**vars(args))
