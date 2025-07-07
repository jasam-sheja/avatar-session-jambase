from torch import Tensor


def relative_kp(
    kp_source: Tensor,
    kp_driving: Tensor,
    kp_driving_reference: Tensor,
    adapt_movement_scale: bool = True,
) -> Tensor:
    """
    Compute relative key points for driving sequence.

    Args:
        kp_source (Tensor): Source key points BxNx2.
        kp_driving (Tensor): Driving key points BxNx2.
        kp_driving_reference (Tensor): Reference driving key points BxNx2.
        _method (str): Method to compute relative key points ('convexhull' or 'bbox').

    Returns:
        Tensor: Relative key points.
    """
    if kp_source.ndim != kp_driving.ndim or kp_source.ndim != kp_driving_reference.ndim:
        raise ValueError(
            f"Key points dimensions do not match: {kp_source.ndim}, {kp_driving.ndim}, {kp_driving_reference.ndim}"
        )
    is_batched = kp_source.ndim == 3
    if not is_batched:
        kp_source = kp_source.unsqueeze(0)
        kp_driving = kp_driving.unsqueeze(0)
        kp_driving_reference = kp_driving_reference.unsqueeze(0)
    if (
        kp_source.shape[1] != kp_driving.shape[1]
        or kp_source.shape[1] != kp_driving_reference.shape[1]
    ):
        raise ValueError(
            f"Key points shapes do not match: {kp_source.shape[1]}, {kp_driving.shape[1]}, {kp_driving_reference.shape[1]}"
        )
    if (
        kp_source.shape[2] != 2
        or kp_driving.shape[2] != 2
        or kp_driving_reference.shape[2] != 2
    ):
        raise ValueError(
            f"Key points must have shape BxNx2: {kp_source.shape[2]}, {kp_driving.shape[2]}, {kp_driving_reference.shape[2]}"
        )
    if adapt_movement_scale:
        # Step 1: compute movement scale
        source_p1 = kp_source.amin(dim=1, keepdim=True)
        source_p2 = kp_source.amax(dim=1, keepdim=True)
        driving_p1 = kp_driving_reference.amin(dim=1, keepdim=True)
        driving_p2 = kp_driving_reference.amax(dim=1, keepdim=True)
        source_area = (source_p2 - source_p1).prod(-1, keepdim=True)
        driving_area = (driving_p2 - driving_p1).prod(-1, keepdim=True)
        adapt_movement_scale = source_area.sqrt() / driving_area.sqrt()
    else:
        adapt_movement_scale = 1.0
    # Step 2: compute relative movement
    kp_value_diff = kp_driving - kp_driving_reference
    kp_value_diff *= adapt_movement_scale
    kp_new = kp_value_diff + kp_source

    if not is_batched:
        return kp_new.squeeze(0)
    else:
        return kp_new
