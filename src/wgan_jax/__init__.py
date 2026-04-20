"""
JAX-native conditional WGAN for tabular data generation in Monte Carlo simulations.

Port of the ds-wgan PyTorch library (Athey, Imbens, Metzger & Munro, 2020)
to JAX/Flax/Optax for efficient GPU training.

Usage:
    from wgan_jax import WGAN, compare_dfs

    wgan = WGAN(
        df=df_train,
        continuous_vars=["age", "income"],
        categorical_vars=["married"],
        conditioning_vars=["treat"],
        continuous_lower_bounds={"age": 0},
    )
    wgan.train(max_epochs=1000, batch_size=256)
    df_synthetic = wgan.generate(df_conditioning)
    compare_dfs(df_train, df_synthetic)
"""

from __future__ import annotations

import functools
import json
import os

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import numpy as np
import optax
import pandas as pd
from flax.training.train_state import TrainState
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# _DataWrapper: pandas <-> JAX conversion
# ---------------------------------------------------------------------------

class _DataWrapper:
    """Handles preprocessing (pandas -> JAX arrays) and deprocessing (JAX -> pandas)."""

    def __init__(
        self,
        df: pd.DataFrame,
        continuous_vars: list[str],
        categorical_vars: list[str],
        conditioning_vars: list[str],
        continuous_lower_bounds: dict[str, float],
        continuous_upper_bounds: dict[str, float],
    ):
        self.variables = dict(
            continuous=list(continuous_vars),
            categorical=list(categorical_vars),
            conditioning=list(conditioning_vars),
        )

        # Continuous statistics
        cont_data = df[continuous_vars].to_numpy(dtype=np.float32) if continuous_vars else np.zeros((len(df), 0), dtype=np.float32)
        self.cont_mean = cont_data.mean(axis=0, keepdims=True)
        self.cont_std = cont_data.std(axis=0, keepdims=True) + 1e-5

        # Conditioning statistics
        cond_data = df[conditioning_vars].to_numpy(dtype=np.float32) if conditioning_vars else np.zeros((len(df), 0), dtype=np.float32)
        self.cond_mean = cond_data.mean(axis=0, keepdims=True)
        self.cond_std = cond_data.std(axis=0, keepdims=True) + 1e-5

        # Categorical metadata
        self.cat_dims = [df[v].nunique() for v in categorical_vars]
        self.cat_labels = [
            np.sort(df[v].unique()).astype(np.float32) for v in categorical_vars
        ]

        # Bounds in normalized space
        lower = np.array(
            [continuous_lower_bounds.get(v, -1e8) for v in continuous_vars],
            dtype=np.float32,
        )
        upper = np.array(
            [continuous_upper_bounds.get(v, 1e8) for v in continuous_vars],
            dtype=np.float32,
        )
        self.cont_bounds = jnp.array(np.stack([
            (lower - self.cont_mean.squeeze()) / self.cont_std.squeeze(),
            (upper - self.cont_mean.squeeze()) / self.cont_std.squeeze(),
        ])) if continuous_vars else jnp.zeros((2, 0))  # shape (2, n_cont)

        # Dimensions
        self.d_cont = len(continuous_vars)
        self.d_cat = sum(self.cat_dims)
        self.d_x = self.d_cont + self.d_cat
        self.d_cond = len(conditioning_vars)

        # Template row for generation
        if continuous_vars or categorical_vars:
            self.df0 = df[continuous_vars + categorical_vars].iloc[0:1].copy()
        else:
            self.df0 = pd.DataFrame()

        # Store one-hot column order from training data for consistent encoding
        if categorical_vars:
            dummy_df = pd.get_dummies(
                df[categorical_vars], columns=categorical_vars
            )
            self._onehot_columns = list(dummy_df.columns)
        else:
            self._onehot_columns = []

    def preprocess(self, df: pd.DataFrame) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert DataFrame to normalized JAX arrays (x, conditioning)."""
        # Continuous
        if self.variables["continuous"]:
            cont = df[self.variables["continuous"]].to_numpy(dtype=np.float32)
            cont = (cont - self.cont_mean) / self.cont_std
        else:
            cont = np.zeros((len(df), 0), dtype=np.float32)

        # Conditioning
        if self.variables["conditioning"]:
            cond = df[self.variables["conditioning"]].to_numpy(dtype=np.float32)
            cond = (cond - self.cond_mean) / self.cond_std
        else:
            cond = np.zeros((len(df), 0), dtype=np.float32)

        # Categorical: one-hot encode
        if self.variables["categorical"]:
            dummy_df = pd.get_dummies(
                df[self.variables["categorical"]],
                columns=self.variables["categorical"],
            )
            # Reindex to match training column order (handles missing categories)
            dummy_df = dummy_df.reindex(columns=self._onehot_columns, fill_value=0)
            cat = dummy_df.to_numpy(dtype=np.float32)
            x = np.concatenate([cont, cat], axis=-1)
        else:
            x = cont

        # NaN check
        combined = np.concatenate([x, cond], axis=-1)
        if np.any(np.isnan(combined)):
            raise ValueError("NaN detected in data after preprocessing.")

        return jnp.array(x), jnp.array(cond)

    def deprocess(
        self, x: jnp.ndarray, conditioning: jnp.ndarray, rng_key: jax.Array
    ) -> pd.DataFrame:
        """Convert JAX arrays back to a DataFrame with original column names and scales."""
        x_np = np.asarray(x)
        cond_np = np.asarray(conditioning)

        # Split continuous / categorical
        cont_np = x_np[:, :self.d_cont]
        cat_logits = x_np[:, self.d_cont:]

        # Denormalize continuous
        cont_np = cont_np * self.cont_std + self.cont_mean

        # Denormalize conditioning
        if self.d_cond > 0:
            cond_np = cond_np * self.cond_std + self.cond_mean

        # Sample categoricals
        cat_values = []
        offset = 0
        for dim, labels in zip(self.cat_dims, self.cat_labels):
            probs = x_np[:, self.d_cont + offset: self.d_cont + offset + dim]
            rng_key, subkey = jax.random.split(rng_key)
            indices = jax.random.categorical(subkey, jnp.log(jnp.array(probs) + 1e-8))
            cat_values.append(labels[np.asarray(indices)].reshape(-1, 1))
            offset += dim

        # Build DataFrame
        all_cols = (
            self.variables["continuous"]
            + self.variables["categorical"]
            + self.variables["conditioning"]
        )
        parts = [cont_np]
        if cat_values:
            parts.append(np.concatenate(cat_values, axis=-1))
        parts.append(cond_np)
        data = np.concatenate(parts, axis=-1)
        return pd.DataFrame(data, columns=all_cols)


# ---------------------------------------------------------------------------
# Flax network definitions
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Conditional generator for tabular data."""
    d_hidden: Sequence[int]
    d_output: int
    d_cont: int
    cat_dims: Sequence[int]
    cont_bounds: jnp.ndarray
    noise_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, conditioning, deterministic=False):
        noise = jax.random.normal(
            self.make_rng("noise"),
            (conditioning.shape[0], self.noise_dim),
        )
        x = jnp.concatenate([noise, conditioning], axis=-1)
        for h in self.d_hidden:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(self.d_output)(x)
        return self._transform(x)

    def _transform(self, x):
        cont = x[:, :self.d_cont]
        cat = x[:, self.d_cont:]
        # Enforce bounds on continuous outputs
        if self.d_cont > 0:
            cont = jnp.clip(cont, self.cont_bounds[0], self.cont_bounds[1])
        # Softmax per categorical group
        if len(self.cat_dims) > 0:
            parts = []
            offset = 0
            for dim in self.cat_dims:
                parts.append(jax.nn.softmax(cat[:, offset:offset + dim], axis=-1))
                offset += dim
            cat = jnp.concatenate(parts, axis=-1)
        return jnp.concatenate([cont, cat], axis=-1)


class Critic(nn.Module):
    """Wasserstein critic for tabular data."""
    d_hidden: Sequence[int]

    @nn.compact
    def __call__(self, x, conditioning):
        h = jnp.concatenate([x, conditioning], axis=-1)
        for d in self.d_hidden:
            h = nn.Dense(d)(h)
            h = nn.relu(h)
        return nn.Dense(1)(h)


# ---------------------------------------------------------------------------
# Gradient penalty
# ---------------------------------------------------------------------------

def _gradient_penalty(
    critic_apply_fn,
    critic_params,
    x_real: jnp.ndarray,
    x_fake: jnp.ndarray,
    conditioning: jnp.ndarray,
    rng_key: jax.Array,
    gp_factor: float,
) -> jnp.ndarray:
    """WGAN-GP gradient penalty (one-sided, matching ds-wgan)."""
    alpha = jax.random.uniform(rng_key, (x_real.shape[0], 1))
    x_interp = alpha * x_real + (1.0 - alpha) * x_fake

    def critic_single(x_i, cond_i):
        out = critic_apply_fn({"params": critic_params}, x_i[None, :], cond_i[None, :])
        return out[0, 0]

    grad_fn = jax.vmap(jax.grad(critic_single, argnums=0))
    gradients = grad_fn(x_interp, conditioning)
    grad_norms = jnp.linalg.norm(gradients, axis=1)
    penalty = jnp.mean(jax.nn.relu(grad_norms - 1.0))
    return gp_factor * penalty


# ---------------------------------------------------------------------------
# Training step factory
# ---------------------------------------------------------------------------

def _make_train_step(
    gen_model: Generator,
    critic_model: Critic,
    gp_factor: float,
    n_critic: int,
    batch_size: int,
):
    """Returns a JIT-compiled train_step function with hyperparams baked in."""

    def _critic_update(carry, _unused):
        """Single critic update, designed for jax.lax.scan."""
        critic_state, gen_params, x_train, cond_train, rng = carry

        # Sample batch (use randint to avoid concretization error with choice)
        rng, batch_rng, noise_rng, gp_rng = jax.random.split(rng, 4)
        idx = jax.random.randint(batch_rng, (batch_size,), 0, x_train.shape[0])
        x_batch = x_train[idx]
        cond_batch = cond_train[idx]

        # Generate fake data (no gradient through generator)
        x_fake = gen_model.apply(
            {"params": gen_params},
            cond_batch,
            deterministic=True,
            rngs={"noise": noise_rng},
        )
        x_fake = jax.lax.stop_gradient(x_fake)

        def critic_loss_fn(critic_params):
            critic_real = critic_model.apply(
                {"params": critic_params}, x_batch, cond_batch
            ).mean()
            critic_fake = critic_model.apply(
                {"params": critic_params}, x_fake, cond_batch
            ).mean()
            wd = critic_real - critic_fake
            gp = _gradient_penalty(
                critic_model.apply, critic_params,
                x_batch, x_fake, cond_batch, gp_rng, gp_factor,
            )
            loss = -wd + gp
            return loss, wd

        (loss, wd), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            critic_state.params
        )
        critic_state = critic_state.apply_gradients(grads=grads)
        new_carry = (critic_state, gen_params, x_train, cond_train, rng)
        return new_carry, wd

    @jax.jit
    def train_step(gen_state, critic_state, x_train, cond_train, rng):
        # --- Critic updates via lax.scan ---
        init_carry = (critic_state, gen_state.params, x_train, cond_train, rng)
        (critic_state, _, _, _, rng), wd_history = jax.lax.scan(
            _critic_update, init_carry, None, length=n_critic,
        )

        # --- Generator update ---
        rng, batch_rng, noise_rng, dropout_rng = jax.random.split(rng, 4)
        idx = jax.random.randint(batch_rng, (batch_size,), 0, x_train.shape[0])
        cond_batch = cond_train[idx]

        def gen_loss_fn(gen_params):
            x_fake = gen_model.apply(
                {"params": gen_params},
                cond_batch,
                deterministic=False,
                rngs={"noise": noise_rng, "dropout": dropout_rng},
            )
            critic_fake = critic_model.apply(
                {"params": critic_state.params}, x_fake, cond_batch,
            ).mean()
            return -critic_fake

        g_loss, grads = jax.value_and_grad(gen_loss_fn)(gen_state.params)
        gen_state = gen_state.apply_gradients(grads=grads)

        metrics = {
            "wd_train": wd_history.mean(),
            "g_loss": g_loss,
        }
        return gen_state, critic_state, metrics, rng

    return train_step


# ---------------------------------------------------------------------------
# WGAN class
# ---------------------------------------------------------------------------

class WGAN:
    """Conditional Wasserstein GAN for tabular data generation.

    Parameters
    ----------
    df : pd.DataFrame
        Training data containing all variables.
    continuous_vars : list[str]
        Continuous variables to generate.
    categorical_vars : list[str]
        Categorical variables to generate.
    conditioning_vars : list[str]
        Variables to condition on (passed through, not generated).
    continuous_lower_bounds : dict
        Lower bounds for continuous variables.
    continuous_upper_bounds : dict
        Upper bounds for continuous variables.
    critic_hidden : list[int]
        Hidden layer sizes for critic.
    generator_hidden : list[int]
        Hidden layer sizes for generator.
    noise_dim : int or None
        Noise dimension (defaults to output dimension).
    generator_dropout : float
        Dropout rate in generator.
    critic_steps : int
        Number of critic updates per generator update.
    critic_lr : float
        Critic learning rate.
    generator_lr : float
        Generator learning rate.
    gp_factor : float
        Gradient penalty coefficient.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        continuous_vars: list[str] | None = None,
        categorical_vars: list[str] | None = None,
        conditioning_vars: list[str] | None = None,
        continuous_lower_bounds: dict[str, float] | None = None,
        continuous_upper_bounds: dict[str, float] | None = None,
        critic_hidden: list[int] | None = None,
        generator_hidden: list[int] | None = None,
        noise_dim: int | None = None,
        generator_dropout: float = 0.1,
        critic_steps: int = 15,
        critic_lr: float = 1e-4,
        generator_lr: float = 1e-4,
        gp_factor: float = 5.0,
        seed: int = 0,
    ):
        continuous_vars = continuous_vars or []
        categorical_vars = categorical_vars or []
        conditioning_vars = conditioning_vars or []
        continuous_lower_bounds = continuous_lower_bounds or {}
        continuous_upper_bounds = continuous_upper_bounds or {}
        critic_hidden = critic_hidden or [128, 128, 128]
        generator_hidden = generator_hidden or [128, 128, 128]

        self._df = df.copy()
        self._dw = _DataWrapper(
            df, continuous_vars, categorical_vars, conditioning_vars,
            continuous_lower_bounds, continuous_upper_bounds,
        )

        d_x = self._dw.d_x
        d_cond = self._dw.d_cond
        noise_dim = noise_dim if noise_dim is not None else d_x

        # Store hyperparameters
        self._critic_steps = critic_steps
        self._gp_factor = gp_factor
        self._generator_dropout = generator_dropout

        # Build models
        self._gen_model = Generator(
            d_hidden=tuple(generator_hidden),
            d_output=d_x,
            d_cont=self._dw.d_cont,
            cat_dims=tuple(self._dw.cat_dims),
            cont_bounds=self._dw.cont_bounds,
            noise_dim=noise_dim,
            dropout_rate=generator_dropout,
        )
        self._critic_model = Critic(d_hidden=tuple(critic_hidden))

        # Initialize parameters
        rng = jax.random.PRNGKey(seed)
        rng, gen_rng, critic_rng = jax.random.split(rng, 3)

        dummy_cond = jnp.ones((2, max(d_cond, 1)))  # batch of 2 for init
        gen_params = self._gen_model.init(
            {"params": gen_rng, "noise": gen_rng, "dropout": gen_rng},
            dummy_cond[:, :d_cond] if d_cond > 0 else dummy_cond[:, :0],
        )["params"]

        dummy_x = jnp.ones((2, max(d_x, 1)))
        critic_params = self._critic_model.init(
            {"params": critic_rng},
            dummy_x[:, :d_x] if d_x > 0 else dummy_x[:, :0],
            dummy_cond[:, :d_cond] if d_cond > 0 else dummy_cond[:, :0],
        )["params"]

        # TrainStates
        self._gen_state = TrainState.create(
            apply_fn=self._gen_model.apply,
            params=gen_params,
            tx=optax.adam(learning_rate=generator_lr),
        )
        self._critic_state = TrainState.create(
            apply_fn=self._critic_model.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        self._rng = rng
        self._trained = False

        print(
            f"WGAN initialized: d_x={d_x} (cont={self._dw.d_cont}, cat={self._dw.d_cat}), "
            f"d_cond={d_cond}, noise_dim={noise_dim}"
        )

    def train(
        self,
        max_epochs: int = 1000,
        batch_size: int = 256,
        test_set_size: int = 64,
        print_every: int = 200,
    ) -> dict[str, list[float]]:
        """Train the WGAN.

        Returns a dict with keys 'wd_train', 'wd_test', 'g_loss' containing
        per-epoch metric histories.
        """
        x, cond = self._dw.preprocess(self._df)
        n = x.shape[0]

        if batch_size > n - test_set_size:
            batch_size = n - test_set_size
            print(f"Reduced batch_size to {batch_size} (dataset has {n} rows, test_set_size={test_set_size})")

        # Train/test split
        rng = self._rng
        rng, split_rng = jax.random.split(rng)
        perm = jax.random.permutation(split_rng, n)
        test_idx, train_idx = perm[:test_set_size], perm[test_set_size:]
        x_train, cond_train = x[train_idx], cond[train_idx]
        x_test, cond_test = x[test_idx], cond[test_idx]

        # Build JIT-compiled train step
        train_step = _make_train_step(
            self._gen_model, self._critic_model,
            self._gp_factor, self._critic_steps, batch_size,
        )

        n_train = x_train.shape[0]
        steps_per_epoch = max(1, n_train // batch_size)

        gen_state = self._gen_state
        critic_state = self._critic_state

        history = {"wd_train": [], "wd_test": [], "g_loss": []}

        pbar = tqdm(range(max_epochs), desc="Training", unit="epoch")
        try:
            for epoch in pbar:
                # Accumulate metrics on-device to avoid per-step GPU sync
                wd_acc = jnp.float32(0.0)
                g_loss_acc = jnp.float32(0.0)
                for _ in range(steps_per_epoch):
                    gen_state, critic_state, metrics, rng = train_step(
                        gen_state, critic_state, x_train, cond_train, rng,
                    )
                    wd_acc = wd_acc + metrics["wd_train"]
                    g_loss_acc = g_loss_acc + metrics["g_loss"]

                # Single host sync per epoch
                wd_train = float(wd_acc) / steps_per_epoch
                g_loss_avg = float(g_loss_acc) / steps_per_epoch

                # Test Wasserstein distance
                wd_test = self._evaluate_wd(self._gen_model, self._critic_model, critic_state, gen_state, x_test, cond_test, rng)
                rng, _ = jax.random.split(rng)

                history["wd_train"].append(wd_train)
                history["wd_test"].append(float(wd_test))
                history["g_loss"].append(g_loss_avg)

                if epoch % print_every == 0:
                    pbar.set_postfix(WD_train=f"{wd_train:.4f}", WD_test=f"{float(wd_test):.4f}")

        except KeyboardInterrupt:
            print("Training interrupted.")

        self._gen_state = gen_state
        self._critic_state = critic_state
        self._rng = rng
        self._trained = True
        return history

    @staticmethod
    @jax.jit
    def _evaluate_wd(gen_model, critic_model, critic_state, gen_state, x_test, cond_test, rng):
        """Compute Wasserstein distance estimate on test data."""
        rng, noise_rng = jax.random.split(rng)
        x_fake = gen_model.apply(
            {"params": gen_state.params},
            cond_test,
            deterministic=True,
            rngs={"noise": noise_rng},
        )
        critic_real = critic_model.apply(
            {"params": critic_state.params}, x_test, cond_test,
        ).mean()
        critic_fake = critic_model.apply(
            {"params": critic_state.params}, x_fake, cond_test,
        ).mean()
        return critic_real - critic_fake

    def generate(
        self,
        df_conditioning: pd.DataFrame | None = None,
        n: int | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic data.

        Parameters
        ----------
        df_conditioning : pd.DataFrame, optional
            DataFrame with conditioning columns. Number of rows determines
            sample size. If None and conditioning_vars is empty, must pass n.
        n : int, optional
            Number of samples to generate. If df_conditioning is None and there
            are no conditioning vars, this is required. If df_conditioning is
            provided, this is ignored.
        seed : int, optional
            Random seed for generation. Uses internal RNG state if not provided.
        """
        rng = jax.random.PRNGKey(seed) if seed is not None else self._rng

        # Build a full DataFrame for preprocessing
        if df_conditioning is not None:
            n_samples = len(df_conditioning)
            df_for_preprocess = df_conditioning.copy()
        elif n is not None:
            n_samples = n
            df_for_preprocess = pd.DataFrame(index=range(n_samples))
        else:
            raise ValueError("Provide df_conditioning or n.")

        # Add placeholder columns for continuous/categorical variables
        updated_cols = self._dw.variables["continuous"] + self._dw.variables["categorical"]
        if self._dw.df0.shape[0] > 0:
            placeholders = self._dw.df0.sample(n_samples, replace=True).reset_index(drop=True)
            for col in updated_cols:
                if col not in df_for_preprocess.columns:
                    df_for_preprocess[col] = placeholders[col].values

        # Add placeholder conditioning columns if needed (for n-only case)
        for col in self._dw.variables["conditioning"]:
            if col not in df_for_preprocess.columns:
                df_for_preprocess[col] = self._df[col].sample(n_samples, replace=True).values

        df_for_preprocess = df_for_preprocess.reset_index(drop=True)

        # Preprocess and generate
        _, cond = self._dw.preprocess(df_for_preprocess)

        rng, noise_rng, sample_rng = jax.random.split(rng, 3)
        x_fake = self._gen_model.apply(
            {"params": self._gen_state.params},
            cond,
            deterministic=True,
            rngs={"noise": noise_rng},
        )

        df_generated = self._dw.deprocess(x_fake, cond, sample_rng)

        # Merge back non-generated columns from input
        gen_cols = set(self._dw.variables["continuous"] + self._dw.variables["categorical"])
        cond_cols = set(self._dw.variables["conditioning"])
        if df_conditioning is not None:
            # Use original conditioning values (not the normalized-then-denormalized ones)
            for col in self._dw.variables["conditioning"]:
                df_generated[col] = df_conditioning[col].values
            # Carry over any extra columns from input
            for col in df_conditioning.columns:
                if col not in gen_cols and col not in cond_cols:
                    df_generated[col] = df_conditioning[col].values

        if seed is None:
            self._rng = rng

        return df_generated

    def save(self, path: str):
        """Save model parameters and metadata to a directory."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "gen_params.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(self._gen_state.params))
        with open(os.path.join(path, "critic_params.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(self._critic_state.params))

        metadata = {
            "continuous_vars": self._dw.variables["continuous"],
            "categorical_vars": self._dw.variables["categorical"],
            "conditioning_vars": self._dw.variables["conditioning"],
            "cat_dims": self._dw.cat_dims,
            "d_cont": self._dw.d_cont,
            "d_cat": self._dw.d_cat,
            "d_cond": self._dw.d_cond,
            "cont_mean": self._dw.cont_mean.tolist(),
            "cont_std": self._dw.cont_std.tolist(),
            "cond_mean": self._dw.cond_mean.tolist(),
            "cond_std": self._dw.cond_std.tolist(),
            "cont_bounds": np.asarray(self._dw.cont_bounds).tolist(),
            "cat_labels": [l.tolist() for l in self._dw.cat_labels],
            "generator_hidden": list(self._gen_model.d_hidden),
            "critic_hidden": list(self._critic_model.d_hidden),
            "noise_dim": self._gen_model.noise_dim,
            "dropout_rate": self._gen_model.dropout_rate,
            "onehot_columns": self._dw._onehot_columns,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved to {path}/")

    @classmethod
    def load(cls, path: str, df: pd.DataFrame | None = None) -> "WGAN":
        """Load a saved WGAN model.

        Parameters
        ----------
        path : str
            Directory where the model was saved.
        df : pd.DataFrame, optional
            Training DataFrame (needed for the df0 template and any further
            training). If not provided, generation still works but train() and
            conditioning-free generate(n=...) may not.
        """
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)

        # Reconstruct DataWrapper manually
        dw = _DataWrapper.__new__(_DataWrapper)
        dw.variables = dict(
            continuous=meta["continuous_vars"],
            categorical=meta["categorical_vars"],
            conditioning=meta["conditioning_vars"],
        )
        dw.cont_mean = np.array(meta["cont_mean"], dtype=np.float32)
        dw.cont_std = np.array(meta["cont_std"], dtype=np.float32)
        dw.cond_mean = np.array(meta["cond_mean"], dtype=np.float32)
        dw.cond_std = np.array(meta["cond_std"], dtype=np.float32)
        dw.cont_bounds = jnp.array(meta["cont_bounds"])
        dw.cat_dims = meta["cat_dims"]
        dw.cat_labels = [np.array(l, dtype=np.float32) for l in meta["cat_labels"]]
        dw.d_cont = meta["d_cont"]
        dw.d_cat = meta["d_cat"]
        dw.d_x = dw.d_cont + dw.d_cat
        dw.d_cond = meta["d_cond"]
        dw._onehot_columns = meta.get("onehot_columns", [])
        if df is not None:
            gen_vars = meta["continuous_vars"] + meta["categorical_vars"]
            dw.df0 = df[gen_vars].iloc[0:1].copy() if gen_vars else pd.DataFrame()
        else:
            dw.df0 = pd.DataFrame()

        # Build models
        gen_model = Generator(
            d_hidden=tuple(meta["generator_hidden"]),
            d_output=dw.d_x,
            d_cont=dw.d_cont,
            cat_dims=tuple(dw.cat_dims),
            cont_bounds=dw.cont_bounds,
            noise_dim=meta["noise_dim"],
            dropout_rate=meta.get("dropout_rate", 0.1),
        )
        critic_model = Critic(d_hidden=tuple(meta["critic_hidden"]))

        # Initialize with dummy data to get param structure
        rng = jax.random.PRNGKey(0)
        rng, gen_rng, critic_rng = jax.random.split(rng, 3)
        d_cond = dw.d_cond
        d_x = dw.d_x

        dummy_cond = jnp.ones((2, max(d_cond, 1)))
        gen_params = gen_model.init(
            {"params": gen_rng, "noise": gen_rng, "dropout": gen_rng},
            dummy_cond[:, :d_cond] if d_cond > 0 else dummy_cond[:, :0],
        )["params"]

        dummy_x = jnp.ones((2, max(d_x, 1)))
        critic_params = critic_model.init(
            {"params": critic_rng},
            dummy_x[:, :d_x],
            dummy_cond[:, :d_cond] if d_cond > 0 else dummy_cond[:, :0],
        )["params"]

        # Load saved params
        with open(os.path.join(path, "gen_params.msgpack"), "rb") as f:
            gen_params = flax.serialization.from_bytes(gen_params, f.read())
        with open(os.path.join(path, "critic_params.msgpack"), "rb") as f:
            critic_params = flax.serialization.from_bytes(critic_params, f.read())

        # Construct WGAN instance without calling __init__
        wgan = cls.__new__(cls)
        wgan._df = df if df is not None else pd.DataFrame()
        wgan._dw = dw
        wgan._gen_model = gen_model
        wgan._critic_model = critic_model
        wgan._critic_steps = 15  # defaults for loaded models
        wgan._gp_factor = 5.0
        wgan._generator_dropout = meta.get("dropout_rate", 0.1)
        wgan._gen_state = TrainState.create(
            apply_fn=gen_model.apply, params=gen_params,
            tx=optax.adam(1e-4),
        )
        wgan._critic_state = TrainState.create(
            apply_fn=critic_model.apply, params=critic_params,
            tx=optax.adam(1e-4),
        )
        wgan._rng = rng
        wgan._trained = True
        print(f"Loaded from {path}/")
        return wgan


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compare_dfs(
    df_real: pd.DataFrame,
    df_fake: pd.DataFrame,
    scatterplot: dict | None = None,
    table_groupby: list[str] | None = None,
    histogram: dict | None = None,
    figsize: int = 3,
):
    """Compare real and generated DataFrames with summary statistics and plots.

    Parameters
    ----------
    df_real, df_fake : pd.DataFrame
    scatterplot : dict, optional
        Keys: 'x' (list of str), 'y' (list of str), 'samples' (int), 'smooth' (float).
    table_groupby : list[str], optional
        Variables to group summary tables by.
    histogram : dict, optional
        Keys: 'variables' (list of str), 'nrow' (int), 'ncol' (int).
    figsize : int
    """
    import matplotlib.pyplot as plt

    table_groupby = table_groupby or []

    df_real = df_real.copy()
    df_fake = df_fake.copy()
    if "source" in df_real.columns:
        df_real = df_real.drop("source", axis=1)
    if "source" in df_fake.columns:
        df_fake = df_fake.drop("source", axis=1)
    df_real.insert(0, "source", "real")
    df_fake.insert(0, "source", "fake")
    common_cols = [c for c in df_real.columns if c in df_fake.columns]
    df_joined = pd.concat(
        [df_real[common_cols], df_fake[common_cols]], axis=0, ignore_index=True
    )
    df_real = df_real.drop("source", axis=1)
    df_fake = df_fake.drop("source", axis=1)
    common_cols = [c for c in df_real.columns if c in df_fake.columns]

    # Mean and std tables
    numeric_cols = df_joined.select_dtypes(include=[np.number]).columns
    groupby_cols = table_groupby + ["source"]
    valid_groupby = [c for c in groupby_cols if c in df_joined.columns]

    means = df_joined[list(set(valid_groupby + list(numeric_cols)))].groupby(valid_groupby).mean().round(2).transpose()
    print("------------- comparison of means -------------")
    print(means)
    print()

    stds = df_joined[list(set(valid_groupby + list(numeric_cols)))].groupby(valid_groupby).std().round(2).transpose()
    print("------------- comparison of stds  -------------")
    print(stds)
    print()

    # Correlation matrices
    numeric_common = [c for c in common_cols if c in df_real.select_dtypes(include=[np.number]).columns]
    if len(numeric_common) > 1:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize * 2, figsize))
        ax1.set_title("real")
        ax2.set_title("fake")
        ax1.matshow(df_real[numeric_common].corr())
        ax2.matshow(df_fake[numeric_common].corr())
        fig1.tight_layout()
        plt.show()

    # Histograms
    if histogram and histogram.get("variables"):
        variables = histogram["variables"]
        nrow = histogram.get("nrow", 1)
        ncol = histogram.get("ncol", len(variables))
        fig2, axes = plt.subplots(nrow, ncol, figsize=(ncol * figsize, nrow * figsize))
        if nrow * ncol == 1:
            axes = np.array([[axes]])
        elif nrow == 1:
            axes = axes[np.newaxis, :]
        elif ncol == 1:
            axes = axes[:, np.newaxis]
        v = 0
        for i in range(nrow):
            for j in range(ncol):
                if v >= len(variables):
                    break
                plot_var = variables[v]
                v += 1
                axes[i][j].hist(
                    [df_real[plot_var], df_fake[plot_var]],
                    bins=20, density=True, histtype="bar",
                    label=["real", "fake"], color=["blue", "red"], alpha=0.7,
                )
                axes[i][j].legend(prop={"size": 8})
                axes[i][j].set_title(plot_var)
        fig2.tight_layout()
        plt.show()

    # Scatterplots
    if scatterplot and scatterplot.get("x") and scatterplot.get("y"):
        x_vars = scatterplot["x"]
        y_vars = scatterplot["y"]
        samples = scatterplot.get("samples", 400)
        smooth = scatterplot.get("smooth", 0)

        n_real = min(samples, len(df_real))
        n_fake = min(samples, len(df_fake))
        df_real_sample = df_real.sample(n_real)
        df_fake_sample = df_fake.sample(n_fake)

        fig3, axes = plt.subplots(
            len(y_vars), len(x_vars),
            figsize=(len(x_vars) * figsize, len(y_vars) * figsize),
            squeeze=False,
        )
        for i, y_var in enumerate(y_vars):
            for j, x_var in enumerate(x_vars):
                ax = axes[i][j]
                xr = df_real_sample[x_var].to_numpy()
                yr = df_real_sample[y_var].to_numpy()
                xf = df_fake_sample[x_var].to_numpy()
                yf = df_fake_sample[y_var].to_numpy()

                if smooth > 0:
                    def _kernel_smooth(xx, yy):
                        xx = (xx - xx.mean()) / (xx.std() + 1e-8)
                        dist = (xx[:, None] - xx[None, :]) ** 2 / (smooth + 1e-9)
                        kern = np.exp(-dist / 2) / np.sqrt(2 * np.pi)
                        w = kern / kern.sum(axis=1, keepdims=True)
                        return w @ yy

                    yr = _kernel_smooth(xr, yr)
                    yf = _kernel_smooth(xf, yf)

                ax.scatter(xr, yr, color="blue", alpha=0.5, s=10, label="real")
                ax.scatter(xf, yf, color="red", alpha=0.5, s=10, label="fake")
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.legend(prop={"size": 8})
        fig3.tight_layout()
        plt.show()
