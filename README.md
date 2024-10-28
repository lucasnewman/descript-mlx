# Descript Audio Codec — MLX

Implementation of the [Descript Audio Codec](https://arxiv.org/abs/2306.06546), with the [MLX](https://github.com/ml-explore/mlx) framework.

Descript can compress 44kHz audio into discrete codes at 8kbps and produces high quality reconstructions at a 90:1 compression ratio compared to the raw audio.

This repository is based on the original Pytorch implementation available [here](https://github.com/descriptinc/descript-audio-codec).

## Installation

```bash
pip install descript-mlx
```

## Usage

You can load a pretrained model from Python like this:

```python
import mlx.core as mx

from descript_mlx import DAC

dac = DAC.from_pretrained("44khz") # or "24khz" / "16khz"
audio = mx.array(...)

# encode into latents and codes
z, codes, latents, commitment_loss, codebook_loss = dac.encode(audio)

# reconstruct from latents/codes to audio
reconstucted_audio = dac.decode(z)

# compress audio to a DAC file
dac_file = dac.compress(audio)
dac_file.save("/path/to/file.dac")

# decompress audio from a DAC file
reconstructed_audio = dac.decompress("/path/to/file.dac")
```

## Citations

```bibtex
@misc{kumar2023highfidelityaudiocompressionimproved,
      title={High-Fidelity Audio Compression with Improved RVQGAN}, 
      author={Rithesh Kumar and Prem Seetharaman and Alejandro Luebs and Ishaan Kumar and Kundan Kumar},
      year={2023},
      eprint={2306.06546},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2306.06546}, 
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
