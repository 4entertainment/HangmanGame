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


# added "#" to "streaming = True" code.
               ```#streaming=True,```


# added "cache_dir="nanoT5/dataset/data" code.
