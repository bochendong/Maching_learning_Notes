# Probabilistic Foundations

## Likelihood function:

If we have independent and identically distributed (i.i.d) data, the probability of seeing all realizations is the product of the probability of each realization

$$
\begin{aligned}
\mathcal{L}(\theta; y_1, y_2, \cdots , y_n) &= \prod_i P_Y(\theta, y_i) \text{ (discrete)} \\
\mathcal{L}(\theta; y_1, y_2, \cdots , y_n) &= \prod_i f_Y(\theta, y_i) \text{ (continuous)}
\end{aligned}
$$

- Log Likelihood:

$$
\begin{aligned}
\mathcal{l}(\theta; y_1, y_2, \cdots , y_n) &= \sum_i \log(P_Y(\theta, y_i)) \text{ (discrete)} \\
\mathcal{l}(\theta; y_1, y_2, \cdots , y_n) &= \sum_i \log(f_Y(\theta, y_i)) \text{ (continuous)}
\end{aligned}
$$

- Normal Log Likelihood:
$$
\begin{aligned}
f_Y(\theta, y) &= \frac{1}{\sqrt{2\pi \sigma_Y^2}}e^{-\frac{(y - \mu)^2}{2 \sigma_Y^2}}\\
\log(f_Y(\theta, y)) &= -\frac{1}{2}\log(2\pi)-\frac{1}{2}\log(\sigma_Y^2) - \frac{(y - \mu_Y)^2}{2\sigma_Y^2}\\
\mathcal{l}(\theta; y_1, y_2, \cdots , y_n) &= -\frac{n}{2}\log(2\pi)-\frac{n}{2}\log(\sigma_Y^2) - \frac{\sum_{i = 1}^n(y - \mu_Y)^2}{2\sigma_Y^2}
\end{aligned}
$$

- Maximum Likelihood Estimation
<div align=center>
    <img src ="MLE.png" width="400" height ="150"/>  
</div>