## Train Keyword Generator (Mask Model)

```bash
python cli.py train --use_keyword=False --model=mask_model
```

## Extract Keywords

```bash
python cli.py extract --model=mask_model
```

## Train Keyword LM

```bash
python cli.py train --model=lstm_keyword_lm --use_keyword=True --learning_rate=1e-04
```
