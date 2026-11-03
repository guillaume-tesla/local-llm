import unittest

import torch

from llm.model import LLM, ModelConfig


def tiny(**kw) -> LLM:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=101, n_layer=2, n_head=4, n_kv_head=kw.pop("n_kv_head", 4),
                      d_model=64, ctx_len=32, **kw)
    return LLM(cfg).eval()


class ModelTests(unittest.TestCase):
    def test_logits_shape_and_loss(self):
        m = tiny()
        x = torch.randint(0, 101, (3, 16))
        logits, loss = m(x, x)
        self.assertEqual(tuple(logits.shape), (3, 16, 101))
        self.assertTrue(torch.isfinite(loss))

    def test_causality(self):
        # Changing a future token must not change predictions at earlier positions.
        for kw in ({}, {"n_kv_head": 2}, {"qk_norm": True}):
            m = tiny(**kw)
            x = torch.randint(0, 101, (1, 20))
            y = x.clone()
            y[0, 12:] = (y[0, 12:] + 7) % 101
            with torch.no_grad():
                a, _ = m(x)
                b, _ = m(y)
            self.assertTrue(torch.allclose(a[0, :12], b[0, :12], atol=1e-5), kw)
            self.assertFalse(torch.allclose(a[0, 12:], b[0, 12:]), kw)

    def test_tied_embeddings_counted_once(self):
        m = tiny()
        self.assertIs(m.lm_head.weight, m.tok_emb.weight)
        self.assertEqual(m.num_params(), sum(p.numel() for p in m.parameters()))

    def test_rejects_too_long_sequence(self):
        m = tiny()
        with self.assertRaises(AssertionError):
            m(torch.zeros(1, 33, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
