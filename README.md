# Adaptive testing moodle experiment

This repository contains minimal logic for adaptive testing using:
- IRT (Rasch Model)
- BKT (Bayesian Knowledge Tracing via pyBKT)

## Structure

- `models/bkt_model.py`: Adaptive BKT model with skill tracking.
- `models/irt_model.py`: Simple IRT model with online theta updating.
- `data/question_metadata.csv`: Question skill and difficulty.
- `data/response_logs.csv`: User answer logs.
- `requirements.txt`: Install dependencies with `pip install -r requirements.txt`.

## Example Usage

```python
from models.bkt_model import BKTModel
from models.irt_model import IRTModel
import pandas as pd

metadata = pd.read_csv("data/question_metadata.csv")

# Initialize and train BKT
bkt = BKTModel(metadata)
bkt.add_response("user1", "q1", 1)
bkt.add_response("user1", "q2", 0)
bkt.train()
print("Weakest skill:", bkt.get_weakest_skill())

# Initialize and update IRT
irt = IRTModel(metadata)
irt.update("q1", 1)
irt.update("q2", 0)
print("Next IRT question:", irt.get_most_informative(["q1", "q2"]))
