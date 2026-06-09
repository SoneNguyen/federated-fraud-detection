# This module defines static exchange rates for various currencies to USD.
# The rates are used by the FX converter to normalize currency values across clients.
# Last updated: 2024-05-13 — all rates to USD
STATIC_RATES = {
    "USD": 1.000,
    "EUR": 1.082,
    "SGD": 0.742,
    "GBP": 1.271,
    "JPY": 0.0067,
    "AUD": 0.655,
}
# Client home currencies — for the federated client prototype
CLIENT_CURRENCIES = {0: "USD", 1: "EUR", 2: "SGD"}