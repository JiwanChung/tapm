# specs

## LSMDC

### Candidates

- Human
- Baseline
- Ours

### Evaluation

For each caption, the workers are advised to score
1(best) to 5(worst) with the criterion of
how much helpful the caption is to a blind person.
We get 200 samples from each candidate,
and collect 3 scores from different workers for each caption.
We compute the median over 3 workers, and average over all captions of a candidate.

Total Works:
- LSMDC: (200 * 3 * 4)

## VIST

### Candidates

- Human
- XE-ss
- AREL
- ***VSCMR: Model***: Deprecated as the rule mining takes too much time
- Ours

### Evaluation

#### Turing Test

The workers are asked to evaluate each machine-generated candidate against the human response.
Given captions from the same photo stream, each worker decide
that caption is better, worse than the gt. There is also an option not to answer in case of unsure decision.
[win, lose, unsure]

#### Comparison Test

We perform pairwise comparisons of our model against other candidiates.
[(ours, XE-ss), (ours, AREL)]
We evaluate on three aspects; expressiveness, concreteness, relevance
The answer choices are same as the turing test.
[win, lose, unsure]

As in the LSMDC evaluation, we get 5 responses for each Comparison and compute the median.
Total Works:
- Turing Test: (150 * 5 * 3)
- Comparison Test: (150 * 5 * 3 * 2)

checkout AMT form in the AREL paper!
