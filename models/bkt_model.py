from pyBKT.models import Model
import pandas as pd

class BKTModel:
    def __init__(self, metadata_df):
        """
        Initializes the pyBKT model with provided metadata.
        metadata_df should contain at least: question_id, skill_name
        """
        self.metadata = metadata_df
        self.model = Model(seed=42, num_fits=1)
        self.trained = False
        self.response_log = pd.DataFrame(columns=["user_id", "question_id", "is_correct"])

    def add_response(self, user_id, question_id, is_correct):
        """
        Appends a response to the internal log.
        This log is used to incrementally train and predict.
        """
        self.response_log = pd.concat([
            self.response_log,
            pd.DataFrame.from_records([{
                "user_id": user_id,
                "question_id": question_id,
                "is_correct": is_correct
            }])
        ], ignore_index=True)

    def train(self):
        """
        Trains the BKT model using the current response log.
        Must be called at least once before predict() or get_weakest_skill().
        """
        data = self.response_log.merge(self.metadata, on="question_id")
        self.model.fit(data)
        self.trained = True

    def predict(self):
        """
        Predicts knowledge state using the most recent response log.
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction.")
        data = self.response_log.merge(self.metadata, on="question_id")
        return self.model.predict(data)

    def get_weakest_skill(self):
        """
        Returns the skill with the lowest current estimated knowledge.
        """
        predictions = self.predict()
        latest = predictions.groupby("skill_name")["p_known"].last()
        return latest.idxmin() if not latest.empty else None
