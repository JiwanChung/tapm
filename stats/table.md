##Ablations

| Model | Features | Keyword | Sampling | Embeddding | CIDEr | Meteor | Bleu4 | RougeL |
| --- | --- |  --- | --- | --- | --- | --- | --- | --- |
| SCN | i3d, resnet | top_1000 | greedy | scratch | 0.1183 | 0.0831 | 0.01166 | 0.1993 |
| SCN | i3d, resnet | subset_1000 | greedy | scratch | 0.1196 | 0.0824 | 0.01209 | 0.2004 |
| SCN | i3d, resnet | complementary_1000 | greedy | scratch | 0.1183 | 0.0831 | 0.01166 | 0.1993 |
| SCN | i3d, resnet | single_1000 | greedy | scratch | 0.1102 | 0.0817 | 0.01174 | 0.2000 |
