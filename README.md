# Modified-ARMA-Model-Implementation-for-Price-Prediction
I implemented an **Adaptive ARMA** (Auto-Regressive Moving Average) model to predict prices of publicly-traded stocks.

While normal ARMA finds values for parameters $\Phi$ and $\theta$ once after iterating through the entire database, here, these parameters were restructured to undergo a stochastic gradient-descent with every new data instance they saw.

This significantly improved prediction accuracy as compared to the ARMA, ARIMA and even SARIMA models, and gave roughly 95% accuracy in testing.

---
**Prediction Rule**

$$
\begin{aligned}
Y_{pred}(t) = \Phi Y(t-1) + \theta\epsilon(t-1) \\
ERROR = Y(t) - Y_{pred}(t) \\
\\
\end{aligned}
$$

**Update Rules**

$$
\begin{aligned}
\Phi = \Phi + (\alpha \cdot ERROR \cdot Y(t-1)) \\
\theta = \theta + (\alpha \cdot ERROR \cdot \epsilon(t-1))
\end{aligned}
$$

---

The predictions from this model are imperfect, since they do not completely account for the everyday volatility of the markets despite being an improvement from other models. The predictions are not financial advice.
