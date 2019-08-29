## Ablations

| Model | dropout | Features | Keyword | Sampling | Embeddding | CIDEr | Meteor | Bleu4 | RougeL |
| --- | --- | --- |  --- | --- | --- | --- | --- | --- | --- |
| SCN | 0.0 | i3d, resnet | top_1000 | greedy | scratch | 0.1183 | 0.0831 | 0.01166 | 0.1993 |
| SCN | 0.0 | i3d, resnet | subset_1000 | greedy | scratch | 0.1196 | 0.0824 | 0.01209 | 0.2004 |
| SCN | 0.5 | i3d, resnet | subset_1000 | greedy | scratch | 0.1042 | 0.0812 | 0.00896 | 0.2021 |
| SCN | 0.0 | i3d, resnet | complementary_1000 | greedy | scratch | 0.1183 | 0.0831 | 0.01166 | 0.1993 |
| SCN | 0.0 | i3d, resnet | single_1000 | greedy | scratch | 0.1102 | 0.0817 | 0.01174 | 0.2000 |
| GPT2 | 0.0 | i3d, resnet | subset_1000 | greedy | GPT2 | 0.1157 | 0.0839 | 0.01131 | 0.2010 |
| GPT2 | 0.5 | i3d, resnet | subset_1000 | greedy | GPT2 | 0.1358 | 0.0852 | 0.01324 | 0.2053 |
| GPT2 | 0.7 | i3d, resnet | subset_1000 | greedy | GPT2 | 0.1391 | 0.0844 | 0.01248 | 0.2048 |
