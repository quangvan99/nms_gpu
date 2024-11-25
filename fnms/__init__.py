try:
    # noinspection PyUnresolvedReferences
    from .gpu_nms import gpu_nms
except ImportError:
    try:
        from .cpu_nms import cpu_nms
    except ImportError:
        from .py_cpu_nms import py_cpu_nms as cpu_nms


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if not force_cpu:
        try:
            return gpu_nms(dets, thresh)
        except NameError:
            return cpu_nms(dets, thresh)
    return cpu_nms(dets, thresh)
