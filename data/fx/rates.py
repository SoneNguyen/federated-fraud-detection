# This module defines static exchange rates for various currencies to USD, 
# which are used in the synthetic data generation process. 
# The rates are based on recent market values and are intended to provide a realistic conversion for the transaction amounts in the generated dataset. 
# The client home currencies are also defined, matching the synthetic data generation setup, to ensure consistency in the currency used for each client's transactions.
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