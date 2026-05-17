# data/fx/rates.py — static prototype rates (production: replace with ECB API)
# Last updated: 2024-05-13 — all rates to USD
STATIC_RATES = {
    "USD": 1.000,
    "EUR": 1.082,
    "SGD": 0.742,
    "GBP": 1.271,
    "JPY": 0.0067,
    "AUD": 0.655,
}
# Client home currencies — matches synthetic data generation
CLIENT_CURRENCIES = {0: "USD", 1: "EUR", 2: "SGD"}