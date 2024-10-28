import unittest

import mlx.core as mx

from einops.array_api import rearrange

from descript_mlx.model import DAC

import soundfile as sf


class TestAll(unittest.TestCase):
    def test_16khz(self):
        model = DAC.from_pretrained(model="16khz")

        audio_data, sample_rate = sf.read("tests/input_16khz.wav")
        audio_data = rearrange(mx.array(audio_data), "n -> 1 1 n")
        self.assertEqual(audio_data.shape, (1, 1, 85325))

        x = model.preprocess(audio_data, sample_rate)
        self.assertEqual(x.shape, (1, 1, 85440))

        z, codes, latents, _, _ = model.encode(x)
        self.assertEqual(z.shape, (1, 1024, 267))
        self.assertEqual(codes.shape, (1, 12, 267))
        self.assertEqual(latents.shape, (1, 96, 267))

        y = model.decode(z)
        self.assertEqual(y.shape, (1, 85432, 1))

    def test_24khz(self):
        model = DAC.from_pretrained(model="24khz")

        audio_data, sample_rate = sf.read("tests/input_24khz.wav")
        audio_data = rearrange(mx.array(audio_data), "n -> 1 1 n")
        self.assertEqual(audio_data.shape, (1, 1, 127987))

        x = model.preprocess(audio_data, sample_rate)
        self.assertEqual(x.shape, (1, 1, 128000))

        z, codes, latents, _, _ = model.encode(x)
        self.assertEqual(z.shape, (1, 1024, 400))
        self.assertEqual(codes.shape, (1, 32, 400))
        self.assertEqual(latents.shape, (1, 256, 400))

        y = model.decode(z)
        self.assertEqual(y.shape, (1, 127992, 1))

    def test_44khz(self):
        model = DAC.from_pretrained(model="44khz")

        audio_data, sample_rate = sf.read("tests/input_44khz.wav")
        audio_data = rearrange(mx.array(audio_data), "n -> 1 1 n")
        self.assertEqual(audio_data.shape, (1, 1, 235176))

        x = model.preprocess(audio_data, sample_rate)
        self.assertEqual(x.shape, (1, 1, 235520))

        z, codes, latents, _, _ = model.encode(x)
        self.assertEqual(z.shape, (1, 1024, 460))
        self.assertEqual(codes.shape, (1, 9, 460))
        self.assertEqual(latents.shape, (1, 72, 460))

        y = model.decode(z)
        self.assertEqual(y.shape, (1, 235520, 1))

    def test_dacfile(self):
        model = DAC.from_pretrained(model="44khz")

        dac_file = model.compress("tests/input_44khz.wav")
        self.assertIsNotNone(dac_file)

        reconstructed = model.decompress(dac_file)
        self.assertIsNotNone(reconstructed)
