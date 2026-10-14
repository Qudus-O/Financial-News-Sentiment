# Financial News Sentiment: Fine-tuned DistilBERT vs. a TF-IDF Baseline


Three-class sentiment classification (Bearish / Bullish / Neutral) of short financial
news tweets. The project measures how much fine-tuning a pretrained transformer
improves on a classical bag-of-words baseline, and where it still fails.

## Results

Test set (2,388 tweets). DistilBERT numbers are the mean of 3 random seeds.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.656 | 0.264 |
| TF-IDF + logistic regression | 0.802 | 0.690 |
| DistilBERT (full fine-tune, 3 epochs) | **0.874** | **0.835** |

Seed-to-seed spread for DistilBERT is about 0.001 macro-F1 (only 3 seeds, so treat
this as approximate).

### Per class

| Class | TF-IDF recall | DistilBERT recall | TF-IDF F1 | DistilBERT F1 |
|---|---|---|---|---|
| Bearish | 0.41 | **0.80** | 0.54 | 0.79 |
| Bullish | 0.56 | 0.80 | 0.66 | 0.81 |
| Neutral | 0.96 | 0.91 | 0.88 | 0.91 |

Macro-F1 is the headline metric because Neutral is about 65% of the data, so accuracy
alone is misleading: always predicting Neutral scores 0.66 accuracy.

## Key findings

- Fine-tuning improves macro-F1 by about 0.145 over TF-IDF, far more than the
  seed-to-seed spread.
- Bearish recall roughly doubles (0.41 to 0.80, seed-to-seed std about 0.014). Bearish
  precision is slightly lower than the baseline's (0.77 vs 0.80).
- The cost is on Neutral: about 9% of Neutral tweets are flagged as Bearish or Bullish,
  up from about 4% with TF-IDF.
- In a random sample of 8 missed Bearish tweets, several describe an event (a probe, a
  data breach) with no negative word. Some apparent errors also sit on arguable labels.
  This is a description of that small sample, not a general claim.
- 38 test texts (1.6%) also appear in the training data. Removing them changes macro-F1
  by about 0.001.

## Method

- **Data:** [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
  (MIT license, version 1.0.0). 9,543 training and 2,388 validation rows as published.
  Labels: 0 = Bearish, 1 = Bullish, 2 = Neutral.
- **Splits:** the published validation set is the **test set**. A stratified 10% of
  training data (955 rows) is a **dev set** used to monitor training. The remaining
  8,588 rows are the **fit set**. Split seed: 42.
- **Baselines:** majority class, and TF-IDF (unigrams and bigrams) + logistic
  regression. Neither baseline was tuned.
- **Model:** `distilbert-base-uncased` with a new 3-class head, trained on the fit set
  for 3 epochs (learning rate 2e-5, batch size 32, max length 64, fp16), with seeds 0, 1
  and 2. Padding is done per batch.

## Data quality

After removing links and lowercasing, 38 of the 2,388 test texts also appear in the
training data, and 106 training rows duplicate other training rows. Of the 38
overlapping test rows, 37 carry a label that also appears for the same text in
training. The notebook reports the full test set plus a check without the overlapping
rows.

## Limitations

- One dataset of short English financial tweets with three classes. Results may not
  transfer to longer documents or other sources.
- The Bearish/Neutral boundary is ambiguous in places, and the labeling process was
  not checked, so some of the measured error is likely label noise.
- Three seeds and one train/dev split, so the reported spread is approximate.
- The choice of 3 epochs was informed by earlier exploratory runs that scored this same
  test split, so the test scores carry a small selection effect.
- The baselines were not tuned, so a tuned baseline would likely score somewhat higher.
- The costs of a missed bearish tweet versus a false alarm depend on the user. Both are
  reported, but neither was optimized.

## How to run

1. Click the Colab badge above (or upload the notebook to Colab).
2. Choose a GPU runtime: Runtime → Change runtime type → T4.
3. Runtime → Run all. The full run takes a few minutes.

Library versions are printed in the Setup section of the notebook. All randomness is
seeded. The tweets are loaded from the Hugging Face Hub at run time and are not stored
in this repository.

## Repository contents

- `finance-sentiment-distilbert.ipynb`: the full analysis, with saved outputs
- `README.md`: this file

## Next steps

- Compare with a finance-pretrained model such as FinBERT.
- Fine-tune with LoRA and compare quality and cost against the full fine-tune.
- Tune the decision threshold for Bearish to trade recall against false alarms.

## License

The dataset is released under the MIT License by its authors.

- Built by Qudus Oseni

- https://linkedin.com/in/qudusoseni82 (LinkedIn)
