from typing import Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange

from descript_mlx.nn.layers import WNConv1d


def normalize(input, p=2.0, dim=1, eps=1e-12):
    norm = mx.power(mx.sum(mx.power(mx.abs(input), p), axis=dim, keepdims=True), 1 / p)
    return input / mx.maximum(norm, eps)


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def __call__(self, z):
        """Quantizes the input tensor using a fixed codebook and returns
        the corresponding codebook vectors.

        Parameters
        ----------
        z : mx.array[B x D x T]

        Returns
        -------
        mx.array[B x D x T]
            Quantized continuous representation of input
        mx.array[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        mx.array[1]
            Codebook loss to update the codebook
        mx.array[B x T]
            Codebook indices (quantized discrete representation of input)
        mx.array[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z.moveaxis(1, 2)).moveaxis(1, 2)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = nn.losses.mse_loss(z_e, z_q, reduction="none").mean([1, 2])
        codebook_loss = nn.losses.mse_loss(z_q, z_e, reduction="none").mean([1, 2])

        z_q = z_e + (
            z_q - z_e
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q.moveaxis(1, 2)).moveaxis(1, 2)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return self.codebook.weight[embed_id]

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).moveaxis(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = normalize(encodings)
        codebook = normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            mx.power(encodings, 2).sum(1, keepdims=True)
            - 2 * encodings @ codebook.T
            + mx.power(codebook, 2).sum(1, keepdims=True).T
        )
        min_dist = (-dist).argmax(1)
        indices = rearrange(min_dist, "(b t) -> b t", b=latents.shape[0])
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = [
            VectorQuantize(input_dim, codebook_size, codebook_dim[i])
            for i in range(n_codebooks)
        ]

    def __call__(self, z, n_quantizers: int = None):
        """Quantizes the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors.

        Parameters
        ----------
        z : mx.array[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
            when in training mode, and a random number of quantizers is used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : mx.array[B x D x T]
                Quantized continuous representation of input
            "codes" : mx.array[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : mx.array[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : mx.array[1]
                Commitment loss to train encoder to predict vectors closer to codebook entries
            "vq/codebook_loss" : mx.array[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            mask = mx.full((z.shape[0],), vals=i) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = mx.stack(codebook_indices, axis=1)
        latents = mx.concatenate(latents, axis=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: mx.array):
        """Given the quantized codes, reconstruct the continuous representation.

        Parameters
        ----------
        codes : mx.array[B x N x T]
            Quantized discrete representation of input

        Returns
        -------
        mx.array[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i.moveaxis(1, 2)).moveaxis(1, 2)
            z_q = z_q + z_q_i
        return z_q, mx.concatenate(z_p, axis=1), codes

    def from_latents(self, latents: mx.array):
        """Given the unquantized latents, reconstruct the continuous representation
        after quantization.

        Parameters
        ----------
        latents : mx.array[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        mx.array[B x D x T]
            Quantized representation of full-projected space
        mx.array[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i.moveaxis(1, 2)).moveaxis(1, 2)
            z_q = z_q + z_q_i

        return z_q, mx.concatenate(z_p, axis=1), mx.stack(codes, axis=1)
