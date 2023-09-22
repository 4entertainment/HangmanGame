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
```#streaming=True,```
\n
-> added "cache_dir="nanoT5/dataset/data" code.


# terminal:

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
Downloading data files:   0%|▋                                                                                                                                                                  | 4/1024 [05:01<21:26:11, 75.66s/it]
Downloading data:  16%|██████████
