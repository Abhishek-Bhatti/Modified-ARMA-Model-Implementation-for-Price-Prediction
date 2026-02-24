# Modified-ARMA-Model-Implementation-for-Price-Prediction
I implemented an **Adaptive ARMA** (Auto-Regressive Moving Average) model to predict prices of publicly-traded stocks.

While normal ARMA finds values for Phi and Eta once after iterating through the entire database, here, these parameters were restructured to undergo a stochastic gradient-descent with every new data instance they saw.

This significantly improved prediction accuracy as compared to the ARMA, ARIMA and even SARIMA models, and gave roughly 95% accuracy in testing.
