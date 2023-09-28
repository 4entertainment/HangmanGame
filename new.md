# New Version:
```
    if args.mode == 'pt':
        local_data_path = 'nanoT5/dataset/data'
        # dataset = datasets.load_dataset(local_data_path,streaming=True)

        with no_ssl_verification():
            dataset = datasets.load_dataset(
                'c4', #20.09
                'en', #20.09
                cache_dir="nanoT5/dataset/data",
                #streaming=True,
            )
```

-> added "#" to "streaming = True" code.
<br/>
```#streaming=True,```
<br/>
-> added "cache_dir="nanoT5/dataset/data" code.


## terminal:
```
(nanoT5) berkin@berkin:~/Desktop/nanoT5$ python -m nanoT5.main optim.name=adafactor optim.lr_scheduler=legacy model.compile=true
Token will not been saved to git credential helper. Pass `add_to_git_credential=True` if you want to set the git credential as well.
Token is valid (permission: read).
Your token has been saved to /home/berkin/.cache/huggingface/token
Login successful
[2023-09-22 15:52:49,681][Main][INFO] - Distributed environment: NO
Num processes: 1
Process index: 0
Local process index: 0
Device: cuda

Mixed precision type: fp16

[2023-09-22 15:52:49,681][Main][INFO] - Working directory is /home/berkin/Desktop/nanoT5/logs/2023-09-22/15-52-49-
loading configuration file config.json from cache at /home/berkin/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/config.json
Model config T5Config {
  "_name_or_path": "google/t5-v1_1-base",
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "transformers_version": "4.33.2",
  "use_cache": true,
  "vocab_size": 32128
}

loading configuration file config.json from cache at /home/berkin/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/config.json
Model config T5Config {
  "_name_or_path": "google/t5-v1_1-base",
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "transformers_version": "4.33.2",
  "use_cache": true,
  "vocab_size": 32128
}

loading file spiece.model from cache at /home/berkin/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/spiece.model
loading file tokenizer.json from cache at None
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at /home/berkin/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/special_tokens_map.json
loading file tokenizer_config.json from cache at /home/berkin/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/tokenizer_config.json
loading configuration file config.json from cache at /home/berkin/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/config.json
Model config T5Config {
  "_name_or_path": "google/t5-v1_1-base",
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "transformers_version": "4.33.2",
  "use_cache": true,
  "vocab_size": 32128
}

You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. If you see this, DO NOT PANIC! This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thouroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
loading configuration file config.json from cache at /home/berkin/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/config.json
Model config T5Config {
  "_name_or_path": "google/t5-v1_1-base",
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "transformers_version": "4.33.2",
  "use_cache": true,
  "vocab_size": 32128
}

[2023-09-22 15:52:53,549][Main][INFO] - You are using T5 legacy LR Schedule, it's independent from the optim.base_lr
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319M/319M [01:05<00:00, 4.89MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 318M/318M [01:17<00:00, 4.13MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 320M/320M [01:11<00:00, 4.49MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319M/319M [01:09<00:00, 4.60MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319M/319M [01:20<00:00, 3.95MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 318M/318M [01:19<00:00, 4.02MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 318M/318M [01:14<00:00, 4.27MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 318M/318M [01:23<00:00, 3.81MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 318M/318M [01:25<00:00, 3.73MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 318M/318M [01:48<00:00, 2.94MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319M/319M [01:18<00:00, 4.08MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 320M/320M [01:30<00:00, 3.55MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 320M/320M [01:20<00:00, 3.97MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319M/319M [02:00<00:00, 2.65MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 320M/320M [01:33<00:00, 3.41MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 318M/318M [01:32<00:00, 3.45MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319M/319M [01:29<00:00, 3.58MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 320M/320M [01:25<00:00, 3.75MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319M/319M [01:20<00:00, 3.98MB/s]
Downloading data files:   2%|███                                                                                                                                                               | 19/1024 [28:08<25:37:35, 91.80s/it]
Downloading data:  71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████
```
## later:
```
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40.5M/40.5M [00:07<00:00, 5.20MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40.4M/40.4M [00:08<00:00, 4.65MB/s]
Downloading data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:29<00:00, 11.18s/it]
Generating train split:   0%|                                                                                                                                                      | 0/364868892 [00:00<?, ? examples/s][2023-09-28 13:03:07,354][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/9224fbb43b687e9227cf354fc3258a0833d6a4bd780ab99584864efbbd919ec3
Generating train split:   0%|▏                                                                                                                                   | 355079/364868892 [00:16<4:38:51, 21785.65 examples/s][2023-09-28 13:03:23,647][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/aefc3cd300d54ed033387a6a8a8dd445b0d0d2ecb898be4bd15d174f565f41b4
Generating train split:   0%|▎                                                                                                                                   | 711375/364868892 [00:32<4:33:21, 22203.22 examples/s][2023-09-28 13:03:39,719][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/1f793794d3a6d7c82a35556c45c2230e6ef8c8adaca05dd2155f483d284da47f
Generating train split:   0%|▍                                                                                                                                  | 1066142/364868892 [00:48<4:30:20, 22428.34 examples/s][2023-09-28 13:03:55,787][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/59dbe05ae8639df7a86cecf2e331cbafcf9aee7550087692040ad28c82708a78
Generating train split:   0%|▌                                                                                                                                  | 1424597/364868892 [01:04<4:27:55, 22607.90 examples/s][2023-09-28 13:04:11,745][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/81ebca015972432d51c7ac7994ed883c2021f5269ef5759b0b7e5d1104dc8f37
Generating train split:   0%|▋                                                                                                                                  | 1778923/364868892 [01:20<4:30:59, 22330.29 examples/s][2023-09-28 13:04:27,825][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/1535b9418e75602286d29c74147902418f6186c766759e9fb1665c59fa59f9ea
Generating train split:   1%|▊                                                                                                                                  | 2134936/364868892 [01:36<4:27:44, 22580.48 examples/s][2023-09-28 13:04:43,855][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/520cce4bc64497bc5daa0bea5d4d4e55b06feeeca5670e4c667daae7a5af8867
Generating train split:   1%|▉                                                                                                                                  | 2490969/364868892 [01:52<4:27:03, 22615.36 examples/s][2023-09-28 13:04:59,864][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/45489bf79bb10a1573057ec1edf3fbf81e1d7ca56afa77ebee4b40cdba47fda9
Generating train split:   1%|█                                                                                                                                  | 2848451/364868892 [02:08<4:25:45, 22702.97 examples/s][2023-09-28 13:05:15,967][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/c60fe8283b86d4197e18164a07dd5d9b69300493963130a3cd6477bc847766ed
Generating train split:   1%|█▏                                                                                                                                 | 3204723/364868892 [02:24<4:27:45, 22511.68 examples/s][2023-09-28 13:05:31,825][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/36709241ab07df987cd42781626f6359f900d33ce5f6d24027209f31b16c8ae3
Generating train split:   1%|█▎                                                                                                                                 | 3561431/364868892 [02:40<4:24:23, 22775.51 examples/s][2023-09-28 13:05:47,888][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/57ff7c96487ae2e0739cc7fb5574861363ae87f19724076238520aa5a75fbce9
Generating train split:   1%|█▍                                                                                                                                 | 3917787/364868892 [02:56<4:23:51, 22799.58 examples/s][2023-09-28 13:06:03,980][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/072875eb39c8238ea7a512b8838a3633f90baeb98bdc0bc217b04fe5923653e4
Generating train split:   1%|█▌                                                                                                                                 | 4275256/364868892 [03:12<5:39:04, 17724.20 examples/s][2023-09-28 13:06:20,041][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/d1796f4880e8d2c08c340abed80f5263426be1daadbfb1fb4361e13a6a051952
Generating train split:   1%|█▋                                                                                                                                 | 4630000/364868892 [03:28<4:27:55, 22409.26 examples/s][2023-09-28 13:06:36,182][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/ea20f029f84baaaffb94317884396db28952d6ab27c91f5091164df536b77713
Generating train split:   1%|█▊                                                                                                                                 | 4988244/364868892 [03:45<4:26:54, 22472.13 examples/s][2023-09-28 13:06:52,379][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/d44fc77c353b1ae4baf934cc916ff773b5cbefd43c92bfffb35150ea9a76ed04
Generating train split:   1%|█▉                                                                                                                                 | 5343623/364868892 [04:01<4:34:07, 21859.28 examples/s][2023-09-28 13:07:08,622][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = nanoT5/dataset/data/downloads/1b9a9a06dfafdb7b8e46849487e5411aafe8be74228f479886380e86bf038411
Generating train split:   2%|██                                                     
```
## then:
```
OSError: [Errno 28] No space left on device 

During handling of the above exception, another exception occurred: 
Traceback (most recent call last): 
File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/builder.py", line 1703, in _prepare_split_single num_examples, num_bytes = writer.finalize() File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/arrow_writer.py", line 586, in finalize self.write_examples_on_file() File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/arrow_writer.py", line 448, in write_examples_on_file self.write_batch(batch_examples=batch_examples) File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/arrow_writer.py", line 559, in write_batch self.write_table(pa_table, writer_batch_size) File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/arrow_writer.py", line 577, in write_table self.pa_writer.write_table(pa_table, writer_batch_size) File "pyarrow/ipc.pxi", line 525, in pyarrow.lib._CRecordBatchWriter.write_table File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/fsspec/implementations/local.py", line 382, in write return self.f.write(*args, **kwargs) 

OSError: [Errno 28] No space left on device 
The above exception was the direct cause of the following exception: 
Traceback (most recent call last): File "/home/berkin/Desktop/nanoT5/nanoT5/main.py", line 46, in main train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args) File "/home/berkin/Desktop/nanoT5/nanoT5/utils/model_utils.py", line 245, in get_dataloaders dataset_splits = load_dataset_splits(args) File "/home/berkin/Desktop/nanoT5/nanoT5/utils/model_utils.py", line 141, in load_dataset_splits dataset = datasets.load_dataset( File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/load.py", line 2153, in load_dataset builder_instance.download_and_prepare( File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/builder.py", line 954, in download_and_prepare self._download_and_prepare( File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/builder.py", line 1717, in _download_and_prepare super()._download_and_prepare( File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/builder.py", line 1049, in _download_and_prepare self._prepare_split(split_generator, **prepare_split_kwargs) File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/builder.py", line 1555, in _prepare_split for job_id, done, content in self._prepare_split_single( File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/builder.py", line 1712, in _prepare_split_single raise DatasetGenerationError("An error occurred while generating the dataset") from e datasets.builder.DatasetGenerationError: An error occurred while generating the dataset 

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```
## another warning from browser:
![Alt text](HangmanGame/1111.png)
