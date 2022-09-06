import torch.nn as nn
from rl.estimators import SharedBiasLinear

from .conv_spectral_norm import spectral_norm_conv2d
from .linear_spectral_norm import spectral_norm


def hook_spectral_normalization(  # pylint: disable=bad-continuation
    spectral,
    layers,
    lipschitz_k=1,
    random_power_iteration=False,
    leave_smaller=False,
    flow_through_norm=False,
):
    """Uses the convention in `spectral` to hook spectral normalization on
    modules in `layers`.

    Args:
        spectral (str):         A string of negative indices. Ex.: `-1` or `-2,-3`.
                                To hook spectral normalization only for computing the
                                norm and not applying it on the weights add
                                the identifier `L`.
                                Ex.: `-1L`, `-2,-3,-4L`.
        layers (list):          Ordered list of tuples of (module_name, nn.Module).
        lipschitz_k (bool):     The target Lipschitz constant.
        random_power_iteration: If True, power iteration is performed by each layer
                                with probability 1/len(normalised_layers).
        leave_smaller (bool):   If False divide by rho, if True by max(rho, 1).

    Returns:
        normalized: Layers
    """
    # Filter unsupported layers
    layers = [
        (n, m)
        for (n, m) in layers
        if isinstance(m, (nn.Conv2d, nn.Linear, SharedBiasLinear))
    ]
    N = len(layers)

    # Some convenient conventions
    if spectral == "":
        # log all layers, but do not apply
        spectral = ",".join([f"-{i}L" for i in range(N)])
    elif spectral == "full":
        # apply snorm everywhere
        spectral = ",".join([f"-{i}" for i in range(N)])
    else:
        spectral = str(spectral)  # sometimes this is just a number eg.: -3

    # For N=5, spectral="-2,-3L":   [('-2', True), ('-3L', False)]
    layers_status = [(i, "L" not in i) for i in spectral.split(",")]
    # For N=5, spectral="-2,-3L":   [(3, True), (2, False)]
    layers_status = [(int(i if s else i[:-1]) % N, s) for i, s in layers_status]

    power_iteration_prob = 1 / len(layers) if random_power_iteration else 1.0
    hooked_layers = []

    for (idx, active) in layers_status:
        layer_name, layer = layers[idx]

        if isinstance(layer, nn.Conv2d):
            spectral_norm_conv2d(
                layer,
                active=active,
                lipschitz_k=lipschitz_k,
                power_iteration_prob=power_iteration_prob,
                leave_smaller=leave_smaller,
                flow_through_norm=flow_through_norm,
            )
        elif isinstance(layer, (nn.Linear, SharedBiasLinear)):
            spectral_norm(
                layer,
                active=active,
                lipschitz_k=lipschitz_k,
                power_iteration_prob=power_iteration_prob,
                leave_smaller=leave_smaller,
                flow_through_norm=flow_through_norm,
            )
        else:
            raise NotImplementedError(
                "S-Norm on {} layer type not implemented for {} @ ({}): {}".format(
                    type(layer), idx, layer_name, layer
                )
            )
        hooked_layers.append((idx, layer))

        print(
            "{} λ={}/p={:2.2f} SNorm to {} @ ({}): {}".format(
                "Active " if active else "Logging",
                lipschitz_k,
                power_iteration_prob,
                idx,
                layer_name,
                layer,
            )
        )
    return hooked_layers
