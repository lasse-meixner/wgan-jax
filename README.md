# wgan-jax

JAX implementation of the conditional Wasserstein GAN for tabular data generation from [Athey, Imbens, Metzger & Munro (2021)](https://doi.org/10.1080/01621459.2021.1893178).

## Setup

```bash
git clone <repo-url>
cd wgan-jax
uv sync
```

## Usage

See `notebooks/` for examples, in particular `lalonde_wgan.ipynb` for the two-stage DS-WGAN on the LaLonde-Dehejia-Wahba dataset.
