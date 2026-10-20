"""Real torch.Module regression for mixed eval/train state after transcription."""
import unittest

import torch
from torch import nn
from hdb2c.nemo_runner import ensure_probe_eval


class ProbeEvalTest(unittest.TestCase):
    def test_eval_parent_with_training_encoder_is_detected_and_repaired(self):
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.pos_enc = nn.Module()
        model.encoder.pos_enc.dropout = nn.Dropout(.1)
        model.encoder.pos_enc.dropout_emb = nn.Dropout(.1)
        model.eval()
        # NeMo's transcription teardown can unfreeze children after parent.eval().
        model.encoder.train()
        self.assertFalse(model.training)
        self.assertTrue(model.encoder.pos_enc.dropout.training)
        proof = ensure_probe_eval(model)
        self.assertFalse(proof['before_eval']['model_training'])
        self.assertTrue(proof['before_eval']['encoder_training'])
        self.assertGreater(proof['before_eval']['training_module_count'], 0)
        self.assertEqual(proof['after_eval']['training_module_count'], 0)
        x = torch.ones((1, 7, 640))
        self.assertTrue(torch.equal(model.encoder.pos_enc.dropout(x), x))
        self.assertTrue(torch.equal(model.encoder.pos_enc.dropout_emb(x), x))
        self.assertEqual(ensure_probe_eval(model)['before_eval']['training_module_count'], 0)

    def test_stubborn_training_child_fails_closed(self):
        class Stubborn(nn.Module):
            def train(self, mode=True):
                self.training = True
                return self
        model = nn.Module()
        model.encoder = Stubborn()
        with self.assertRaisesRegex(RuntimeError, 'still has training modules'):
            ensure_probe_eval(model)


if __name__ == '__main__':
    unittest.main()
