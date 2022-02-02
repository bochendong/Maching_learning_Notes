# Optimality and Improvement

# Bellman Expectation Equations

$$
\begin{aligned}
    v(s) &= \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t = s]\\

    q(s,a) &= \mathbb{E}[R_{t+1} + \gamma q(S_{t+1}, A_{t+1}) | S_t = s, A_t = a ]
\end{aligned}
$$

$$
\begin{aligned}
v_{\pi}(s) &= \sum \pi(a|s) \cdot q_{\pi}(s,a)\\
q_{\pi}(s, a) &= \sum  p(s^{\prime}, r| s, a) \cdot [r + \gamma v_\pi(s^{\prime})] \\
v_{\pi}(s) &= \sum \pi(a|s) \cdot \sum p(s^{\prime}, r| s, a) \cdot [r + \gamma v_\pi(s^{\prime})] \\
q_{\pi}(s, a) &= \sum p(s^{\prime}| s,a) \cdot [r_{s^{\prime}, a} + \gamma \sum\pi(a^\prime | s^\prime) \cdot q_\pi(s^\prime, a^\prime)]
\end{aligned}
$$