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
