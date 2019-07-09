import numpy as np

def preprocess_rates(rates):
    length = rates.shape[0]
    percent = int(0.05 * length)

    sorted_rates = np.sort(rates)
    valid_rates = sorted_rates[percent:length - percent]

    return valid_rates

class Team():
    def __init__(self, name, rates):
        self.name = name
        self.rates = preprocess_rates(rates)
        self.mean_rate = self.rates.mean()
        self.median_rate = np.median(self.rates)
        self.RH = 0  # coef for Home matches
        self.RA = 0