```bash
cd scripts
```

## get top-K words **by frequency**

```bash
python get_topk_keywords.py --n=K
```

## get top-K words **by Rake**

```bash
python get_rake_keywords.py --n=K
```

## get top-K words **by Chosen Model**

```bash
python cli.py extract --ckpt_name=CKPT_NAME
python postprocess_keyword.py --n=K --path=KEYWORD_DIR -f=full
```

The above commands will produce respective keyword files in `DATA_DIR/keywords` folder.

## compare stats between given keywords

```
python compare_keywords.py
```

The above command will print stats for every keyword file in `DATA_DIR/keywords` folder.
