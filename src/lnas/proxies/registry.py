from lnas.config import ProxyConfig

from .activation import ActivationKernelProxy
from .flops import FLOPsProxy
from .lle import LLEProxy


def build_proxy(config: ProxyConfig):
    name = config.name.lower()
    if name == "lle":
        return LLEProxy(config.max_outputs, config.epsilon, config.stable_only)
    if name == "hd":
        return ActivationKernelProxy(False, config.epsilon)
    if name == "sahd":
        return ActivationKernelProxy(True, config.epsilon)
    if name == "flops":
        return FLOPsProxy()
    raise ValueError("proxy.name must be one of: lle, hd, sahd, flops")
