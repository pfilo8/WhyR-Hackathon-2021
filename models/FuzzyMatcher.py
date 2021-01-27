import re

import numpy as np

from fuzzywuzzy import fuzz


class FuzzyMatcher:
    def __init__(self):
        pass

    @staticmethod
    def extract_year(string):
        return np.max(np.array(re.findall(r'(\d{4})', string)).astype(int))

    @staticmethod
    def extract_venue_code(string):
        return re.findall(r'[a-d]{10}', string)[0]

    def predict_proba(self, x, y):
        out = []
        assert len(x) == len(y)
        for sent_x, sent_y in zip(x, y):
            year_x = self.extract_year(sent_x)
            year_y = self.extract_year(sent_y)
            code_x = self.extract_venue_code(sent_x)
            code_y = self.extract_venue_code(sent_y)

            pred = fuzz.token_set_ratio(sent_x, sent_y)
            pred -= 50 if year_x != year_y else 0
            pred -= 50 if code_x != code_y else 0
            if pred < 0:
                pred = 0
            out.append(pred/100)
        return np.array(out)

    def predict(self, x, y, threshold=0.5):
        out = self.predict_proba(x, y)
        return (out > threshold).astype(int)
