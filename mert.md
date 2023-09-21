(nanoT5) berkin@berkin:~/Desktop/nanoT5$ python -m nanoT5.main optim.name=adafactor optim.lr_scheduler=legacy model.compile=true
[2023-09-21 15:03:38,161][Main][INFO] - Distributed environment: NO
Num processes: 1
Process index: 0
Local process index: 0
Device: cuda

Mixed precision type: fp16

[2023-09-21 15:03:38,162][Main][INFO] - Working directory is /home/berkin/Desktop/nanoT5/logs/2023-09-21/15-03-38-
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

[2023-09-21 15:03:41,763][Main][INFO] - You are using T5 legacy LR Schedule, it's independent from the optim.base_lr
Using the latest cached version of the module from /home/berkin/.cache/huggingface/modules/datasets_modules/datasets/c4/df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01 (last modified on Tue Sep 19 15:47:23 2023) since it couldn't be found locally at c4., or remotely on the Hugging Face Hub.
[2023-09-21 15:03:43,998][datasets.load][WARNING] - Using the latest cached version of the module from /home/berkin/.cache/huggingface/modules/datasets_modules/datasets/c4/df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01 (last modified on Tue Sep 19 15:47:23 2023) since it couldn't be found locally at c4., or remotely on the Hugging Face Hub.
[2023-09-21 15:03:45,307][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00890-of-01024.json.gz
[2023-09-21 15:03:45,307][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00427-of-01024.json.gz
[2023-09-21 15:03:45,307][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.01015-of-01024.json.gz
[2023-09-21 15:03:45,307][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00720-of-01024.json.gz
[2023-09-21 15:03:45,308][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00654-of-01024.json.gz
[2023-09-21 15:03:45,308][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00821-of-01024.json.gz
[2023-09-21 15:03:45,309][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00011-of-01024.json.gz
[2023-09-21 15:03:45,309][datasets_modules.datasets.c4.df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01.c4][INFO] - generating examples from = https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00174-of-01024.json.gz
Error executing job with overrides: ['optim.name=adafactor', 'optim.lr_scheduler=legacy', 'model.compile=true']
Traceback (most recent call last):
  File "/home/berkin/Desktop/nanoT5/nanoT5/main.py", line 68, in main
    train(model, train_dataloader, test_dataloader, accelerator,
  File "/home/berkin/Desktop/nanoT5/nanoT5/utils/train_utils.py", line 186, in train
    for batch_id, batch in enumerate(train_dataloader, start=1):
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/accelerate/data_loader.py", line 560, in __iter__
    next_batch, next_batch_info = self._fetch_batches(main_iterator)
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/accelerate/data_loader.py", line 523, in _fetch_batches
    batches.append(next(iterator))
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 633, in __next__
    data = self._next_data()
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1345, in _next_data
    return self._process_data(data)
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1371, in _process_data
    data.reraise()
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/torch/_utils.py", line 644, in reraise
    raise exception
FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 32, in fetch
    data.append(next(self.dataset_iter))
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 1358, in __iter__
    yield from self._iter_pytorch()
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 1293, in _iter_pytorch
    for key, example in ex_iterable:
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 982, in __iter__
    for x in self.ex_iterable:
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 678, in __iter__
    yield from self._iter()
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 693, in _iter
    for key, example in iterator:
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 1114, in __iter__
    for key, example in self.ex_iterable:
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 678, in __iter__
    yield from self._iter()
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 740, in _iter
    for key, example in iterator:
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 1114, in __iter__
    for key, example in self.ex_iterable:
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/iterable_dataset.py", line 233, in __iter__
    yield from self.generate_examples_fn(**self.kwargs)
  File "/home/berkin/.cache/huggingface/modules/datasets_modules/datasets/c4/df532b158939272d032cc63ef19cd5b83e9b4d00c922b833e4cb18b2e9869b01/c4.py", line 88, in _generate_examples
    with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/streaming.py", line 74, in wrapper
    return function(*args, download_config=download_config, **kwargs)
  File "/home/berkin/anaconda3/envs/nanoT5/lib/python3.8/site-packages/datasets/download/streaming_download_manager.py", line 507, in xopen
    raise FileNotFoundError(
FileNotFoundError: https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.00821-of-01024.json.gz
If the repo is private or gated, make sure to log in with `huggingface-cli login`.


Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
