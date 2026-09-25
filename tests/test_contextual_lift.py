import unittest

import torch

from BCW.bcw import BCW
from BCW.encoder import ContextualLift


class ContextualLiftTests(unittest.TestCase):
    def test_context_state_is_fixed_size_and_updates_between_chunks(self) -> None:
        module = ContextualLift(dim=3)
        first = torch.randn(2, 5, 3)
        second = torch.randn(2, 2, 3)

        _, _, mask, state = module(first)
        _, _, _, next_state = module(second, state)

        self.assertEqual(mask.shape, first.shape)
        self.assertEqual(state.shape, (2, 3))
        self.assertEqual(next_state.shape, (2, 3))
        self.assertFalse(torch.equal(state, next_state))

    def test_bcw_chunked_path_backpropagates_to_context_primitive(self) -> None:
        torch.manual_seed(0)
        model = BCW(vocab_size=16, ctx_length=8, d=3, chunk_size=2)
        patches = torch.randint(0, 16, (2, 8))

        gated, predicted, losses = model(patches)
        model.zero_grad(set_to_none=True)
        model.backward_loss(
            patches, lambda_r1=1.0, lambda_compress=0.1,
            compression_scale=torch.tensor(1.0))

        context = model.encoder.contextual_lift
        assert context is not None
        self.assertEqual(gated.shape, (2, 8, 3))
        self.assertEqual(predicted.shape, (2, 8))
        self.assertIsNotNone(context.state_update.weight_ih.grad)

        omitted, _, _ = model(patches, return_gated=False)
        self.assertEqual(omitted.numel(), 0)


if __name__ == "__main__":
    unittest.main()
