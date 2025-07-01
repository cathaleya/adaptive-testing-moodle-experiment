# Adaptive testing moodle experiment

#Repositori ini berisi logika minimal untuk pengujian adaptif menggunakan:
#IRT (Model Rasch)
#BKT (Bayesian Knowledge Tracing melalui pyBKT)

#Struktur
#models/bkt_model.py: Model BKT adaptif dengan pelacakan keterampilan.
#models/irt_model.py: Model IRT sederhana dengan pembaruan theta secara online.
#data/question_metadata.csv: Metadata soal, berisi keterampilan dan tingkat kesulitan soal.
#data/response_logs.csv: Log jawaban pengguna.
#requirements.txt: Instal dependensi dengan menjalankan pip install -r requirements.txt.


## Contoh Penggunaan

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
