"""B8 extension of native WAM; the v1/v2 adapter remains unchanged."""
from runtime.wan.native_baseline_models_v1 import NativeAdapter, finite_tensor, quantize


def parse_wam(preds, native_message):
    """Only predictions and the native parser. No query, ROI truth or editor state."""
    import numpy as np
    import torch
    b = len(preds)
    finite_tensor(preds, (b, 33, 256, 256))
    mask = preds[:, :1].sigmoid()
    selected = mask > .5
    native = native_message(preds[:, 1:], mask, method='semihard')
    counts = selected.sum(dim=(1, 2, 3))
    sums = (preds[:, 1:] * selected).double().sum(dim=(2, 3))
    # Public fixed geometry statistic, never passed into the native parser.
    center = selected[:, :, 64:192, 64:192].sum(dim=(1, 2, 3))
    valid = counts > 0
    means = torch.zeros((b, 32), dtype=torch.float64, device=preds.device)
    for i in range(b):
        if valid[i]:
            means[i] = preds[i, 1:].masked_select(selected[i].expand_as(preds[i, 1:])).reshape(32, -1).mean(-1, dtype=torch.float64)
    return dict(bits=native.to(torch.uint8).cpu().numpy(), valid=valid.cpu().numpy(),
                selected_count=counts.cpu().numpy().astype(np.uint32),
                center_count=center.cpu().numpy().astype(np.uint32),
                sums=sums.cpu().numpy(), means=means.cpu().numpy(),
                mask_shape=np.asarray([256, 256], dtype=np.int16))


class NativeAdapterV3(NativeAdapter):
    def embed(self, rgb, message, rectangle):
        if self.method != 'wam':
            return super().embed(rgb, message, rectangle)
        import torch
        if not 1 <= len(rgb) <= 8:
            raise ValueError('WAM v3 requires a real tail or fixed B8')
        with torch.inference_mode():
            x = self.tensor(rgb)
            msg = torch.tensor([message]*len(rgb), dtype=torch.float32, device=self.device)
            y = self.unnormalize(self.model.embed(self.normalize(x), msgs=msg)['imgs_w'])
            finite_tensor(y, x.shape)
            y0, x0, y1, x1 = rectangle
            composed = x.clone()
            composed[:, :, y0:y1, x0:x1] = y[:, :, y0:y1, x0:x1]
            return quantize(composed)

    def detect(self, rgb):
        if self.method != 'wam':
            return super().detect(rgb)
        import torch
        if not 1 <= len(rgb) <= 8:
            raise ValueError('WAM v3 requires a real tail or fixed B8')
        with torch.inference_mode():
            x = self.tensor(rgb)
            return parse_wam(self.model.detect(self.normalize(x))['preds'], self.native_message)
