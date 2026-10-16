# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")
# Another AIM's tests may leave a bare stub module registered as torch in the shared session.
if getattr(torch, "__file__", None) is None:
    pytest.skip("torch is a stub module, not the installed package", allow_module_level=True)

from dinov2_inference import Dinov2Predictor  # noqa: E402


def test_preprocess_outputs_expected_tensor_shape_and_range():
    image = Image.fromarray(np.full((32, 40, 3), 127, dtype=np.uint8), mode="RGB")
    predictor = Dinov2Predictor()

    tensor = predictor._preprocess(image)

    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert str(tensor.dtype) == "torch.float32"
    assert float(tensor.abs().max()) < 3.0


def test_preprocess_batch_outputs_expected_tensor_shape():
    images = [Image.fromarray(np.full((32, 40, 3), value, dtype=np.uint8), mode="RGB") for value in (64, 192)]
    predictor = Dinov2Predictor()

    tensor = predictor._preprocess_batch(images)

    assert tuple(tensor.shape) == (2, 3, 224, 224)
    assert str(tensor.dtype) == "torch.float32"


def test_preprocess_preserves_aspect_ratio_and_center_crops():
    pixels = np.zeros((200, 100, 3), dtype=np.uint8)
    pixels[:50] = (255, 0, 0)
    pixels[50:150] = (0, 255, 0)
    pixels[150:] = (0, 0, 255)
    predictor = Dinov2Predictor()

    tensor = predictor._preprocess(Image.fromarray(pixels, mode="RGB"))

    restored = tensor[0].numpy().transpose(1, 2, 0)
    restored = restored * np.asarray((0.229, 0.224, 0.225)) + np.asarray((0.485, 0.456, 0.406))
    assert np.allclose(restored, (0.0, 1.0, 0.0), atol=1e-6)
