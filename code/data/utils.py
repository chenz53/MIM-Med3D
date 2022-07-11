from typing import Any, Callable, List, Sequence, Tuple, Union

import glob
import numpy as np
import kornia.augmentation as K
from einops import rearrange, repeat

import torch
import torch.nn.functional as F

from monai.data.utils import (
    compute_importance_map,
    dense_patch_slices,
    get_valid_patch_size,
)
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
from monai.transforms import MapTransform


def list_splitter(list_to_split, ratio):
    first_half = int(len(list_to_split) * ratio)

    return list_to_split[:first_half], list_to_split[first_half:]


def get_modalities(path_to_files: list) -> dict:
    all_files = glob.glob(path_to_files + "/*nii.gz")
    if len(all_files) != 5:
        raise ValueError(f"Number of files are not equal to 5 in {path_to_files}")
    modality_finder = lambda x: x.split("/")[-1].split("_")[-1].split(".")[0]

    return {modality_finder(x): x for x in all_files}


class StackStuff(MapTransform):
    def __call__(self, data):
        d = dict(data)
        list_of_keys = ["flair", "t1ce", "t1", "t2"]
        list_of_images = [
            torch.from_numpy(data[x].astype(np.float32)) for x in list_of_keys
        ]
        d["image"] = torch.stack(list_of_images)
        for key in list_of_keys:
            d.pop(key)

        return d


class StackStuffM(MapTransform):
    def __call__(self, data):
        d = dict(data)
        list_of_keys = ["flair", "t1ce", "t1", "t2", "Mask"]
        list_of_images = [
            torch.from_numpy(data[x].astype(np.float32)) for x in list_of_keys
        ]
        mask = data["Mask"].astype(bool)
        for index in range(len(list_of_images)):
            list_of_images[index][~mask] = 0
        d["image"] = torch.stack(list_of_images[:-1])
        for key in list_of_keys[1:]:
            d.pop(key)

        return d


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 4, d[key] == 1))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(np.logical_or(d[key] == 1, d[key] == 2), d[key] == 4)
            )
            # label 2 is ET
            result.append(d[key] == 4)
            d["label"] = np.stack(result, axis=0).astype(np.float32)
        return d


class DataAugmentation(torch.nn.Module):
    def __init__(
        self, flip_probability=0.5, affine_probability=0.5,
    ):
        super(DataAugmentation, self).__init__()
        self.flip_probability = flip_probability
        self.affine_probability = affine_probability
        self.flip_0 = K.RandomDepthicalFlip3D(p=self.flip_probability)
        self.flip_1 = K.RandomHorizontalFlip3D(p=self.flip_probability)
        self.flip_2 = K.RandomVerticalFlip3D(p=self.flip_probability)

        affine_args = {
            "degrees": 10,
            "translate": None,
            "scale": (1.15, 0.85),
            "shears": 5,
            "p": self.affine_probability,
        }

        self.image_affine = K.RandomAffine3D(resample="BILINEAR", **affine_args)
        self.mask_affine = K.RandomAffine3D(resample="NEAREST", **affine_args)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    @torch.no_grad()
    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for flip_index in range(3):
            flip = getattr(self, f"flip_{flip_index}")
            img = flip(img)
            mask = flip(mask, flip._params)

        img = self.image_affine(img)
        mask = self.mask_affine(mask, self.image_affine._params)
        return img, mask


def sliding_window_reconstruction(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    reshape: bool = True,
    patch_size: Sequence[int] = [16, 16, 16],
    masked_value: int = 0,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window reconstruction on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(
        inputs,
        pad=pad_size,
        mode=look_up_option(padding_mode, PytorchPadMode).value,
        value=cval,
    )

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size),
        mode=mode,
        sigma_scale=sigma_scale,
        device=device,
    )

    # Perform predictions
    output_image, count_map = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    masked_image = torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
            sw_device
        )
        # if using MAE
        if reshape:
            pred_pixel_values, patches, batch_range, masked_indices = predictor(
                window_data, *args, **kwargs
            )
            # batch, num_patches, dim = patches.shape
            # masked_patches = patches.clone()
            # masked_patches[batch_range, masked_indices] = masked_value
            # mask_tokens = repeat(
            #     torch.tensor([1.0], device=device),
            #     "1 -> b n d",
            #     b=batch,
            #     n=num_patches,
            #     d=dim,
            # )
            # masked_bool_mask = (
            #     torch.zeros((batch, num_patches), device=device)
            #     .scatter_(-1, masked_indices, 1)
            #     .bool()
            # )
            # # mask tokens
            # masked_patches = torch.where(
            #     masked_bool_mask[..., None], mask_tokens, patches
            # )
            patches[batch_range, masked_indices] = masked_value
            mask_prob = rearrange(
                patches,
                "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)",
                h=roi_size[0] // patch_size[0],
                w=roi_size[1] // patch_size[1],
                d=roi_size[2] // patch_size[2],
                c=inputs.shape[1],
                p1=patch_size[0],
                p2=patch_size[1],
                p3=patch_size[2],
            )
            patches[batch_range, masked_indices] = pred_pixel_values
            seg_prob = rearrange(
                patches,
                "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)",
                h=roi_size[0] // patch_size[0],
                w=roi_size[1] // patch_size[1],
                d=roi_size[2] // patch_size[2],
                c=inputs.shape[1],
                p1=patch_size[0],
                p2=patch_size[1],
                p3=patch_size[2],
            )
        else:
            seg_prob = predictor(window_data, *args, **kwargs).to(
                device
            )  # batched patch segmentation

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            # masked image reconstruction
            masked_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            masked_image[original_idx] += importance_map * mask_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map
    masked_image = masked_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(
            pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2]
        )
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing], masked_image[final_slicing]


def sliding_window_embedding(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(
        inputs,
        pad=pad_size,
        mode=look_up_option(padding_mode, PytorchPadMode).value,
        value=cval,
    )

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size),
        mode=mode,
        sigma_scale=sigma_scale,
        device=device,
    )

    # Perform predictions
    output_image, count_map = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    _initialized = False
    res = []
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
            sw_device
        )
        seg_prob, _ = predictor(
            window_data, *args, **kwargs
        )  # batched patch segmentation
        seg_prob = seg_prob.to(sw_device)
        res.append(seg_prob)

    #     if not _initialized:  # init. buffer at the first iteration
    #         output_classes = seg_prob.shape[1]
    #         output_shape = [batch_size, output_classes] + list(image_size)
    #         # allocate memory to store the full output and the count for overlapping parts
    #         output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
    #         count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
    #         _initialized = True

    #     # store the result in the proper location of the full output. Apply weights from importance map.
    #     for idx, original_idx in zip(slice_range, unravel_slice):
    #         output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
    #         count_map[original_idx] += importance_map

    # # account for any overlapping sections
    # output_image = output_image / count_map

    # final_slicing: List[slice] = []
    # for sp in range(num_spatial_dims):
    #     slice_dim = slice(
    #         pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2]
    #     )
    #     final_slicing.insert(0, slice_dim)
    # while len(final_slicing) < len(output_image.shape):
    #     final_slicing.insert(0, slice(None))
    # return output_image[final_slicing]
    return res


def _get_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int,
    overlap: float,
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


if __name__ == "__main__":
    from monai.data.utils import dense_patch_slices

    slices = dense_patch_slices(
        image_size=(224, 224, 100), patch_size=(16, 16, 16), scan_interval=(16, 16, 16),
    )

    print(slices)
