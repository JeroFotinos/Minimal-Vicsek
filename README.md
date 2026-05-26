# Vicsek Model

Minimal local implementation of the two-dimensional Vicsek model, based on the model introduced by Vicsek *et al.* (1995).

## Model

For particle \(i\),

$$ \theta_i(t+\Delta t) =
\arg\left[
\sum_{j: |\mathbf{x}_j-\mathbf{x}_i| \le r}
e^{i\theta_j(t)}
\right]
+
\xi_i(t),
$$

$$ \mathbf{x}_i(t+\Delta t) =
\mathbf{x}_i(t)
+
v\,\Delta t
\begin{pmatrix}
\cos\theta_i(t+\Delta t) \\
\sin\theta_i(t+\Delta t)
\end{pmatrix},
$$

with periodic boundary conditions. The angular noise is sampled from

$$
\xi_i(t) \sim \mathcal{U}\left(-\frac{\eta}{2}, \frac{\eta}{2}\right).
$$

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python minimal_vicsek.py
```

Edit the parameters at the bottom of `minimal_vicsek.py` to explore different regimes.

## Reference

[Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I., & Shochet, O. (1995). Novel type of phase transition in a system of self-driven particles. *Physical Review Letters*, 75(6), 1226–1229.](https://web.mit.edu/~jadbabai/www/ESE680/Vicsek_SPP.pdf)
