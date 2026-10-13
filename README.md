# Sonar Rock vs Mine Prediction

Binary classification of sonar returns as rock (`R`) or mine (`M`) using Logistic Regression, with a
minimal Flask endpoint that serves the trained model. Hands-on learning project.

## Dataset

`data/sonar.csv` - the UCI "Connectionist Bench (Sonar, Mines vs. Rocks)" dataset: 208 samples,
60 numeric features (energy in frequency bands, 0-1), label in column 60 (`R` or `M`). No header row.

## Approach (`sonar_pred.ipynb`)

- 80/20 train/test split (`random_state=1`), default `LogisticRegression`.
- Evaluated with accuracy; the trained model is pickled to `model.pkl`.

Results printed in the notebook:

| Split | Accuracy |
|-------|----------|
| Train | 0.8313 |
| Test  | 0.8095 |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook sonar_pred.ipynb      # training / exploration
python app.py                          # Flask dev server on http://127.0.0.1:5000
```

Predict with the API by POSTing the 60 feature values:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"data": [0.0164,0.0173,0.0347,0.0070,0.0187,0.0671,0.1056,0.0697,0.0962,0.0251,0.0801,0.1056,0.1266,0.0890,0.0198,0.1133,0.2826,0.3234,0.3238,0.4333,0.6068,0.7652,0.9203,0.9719,0.9207,0.7545,0.8289,0.8907,0.7309,0.6896,0.5829,0.4935,0.3101,0.0306,0.0244,0.1108,0.1594,0.1371,0.0696,0.0452,0.0620,0.1421,0.1597,0.1384,0.0372,0.0688,0.0867,0.0513,0.0092,0.0198,0.0118,0.0090,0.0223,0.0179,0.0084,0.0068,0.0032,0.0035,0.0056,0.0040]}'
# -> {"output": "R"}
```

(The sample is the one used in the notebook, which predicts `R`.)

## Project structure

```
sonar_pred.ipynb   # data loading, training, evaluation, pickling
app.py             # Flask API: GET / and POST /predict
model.pkl          # trained LogisticRegression
data/sonar.csv
```
