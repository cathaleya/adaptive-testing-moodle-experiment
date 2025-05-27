class IRTModel:
    def __init__(self, metadata_df):
        """
        Initializes a minimal Rasch-based IRT model.
        """
        self.theta = 0.0
        self.difficulties = metadata_df.set_index("question_id")["difficulty"].to_dict()

    def update(self, question_id, is_correct):
        """
        Updates theta based on the latest response.
        """
        b = self.difficulties.get(question_id, 0.0)
        lr = 0.1
        p = 1 / (1 + 2.718 ** (-(self.theta - b)))
        self.theta += lr * (is_correct - p)

    def get_most_informative(self, asked_questions):
        """
        Returns the question ID with the highest Fisher information.
        """
        def info(b):  # Fisher information function
            p = 1 / (1 + 2.718 ** (-(self.theta - b)))
            return p * (1 - p)

        unasked = {q: b for q, b in self.difficulties.items() if q not in asked_questions}
        return max(unasked, key=lambda q: info(unasked[q]), default=None)
