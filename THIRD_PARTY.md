# Third-party references

This implementation uses the published search-space definitions and documented interfaces of the following projects. Their repositories remain the authoritative sources for the original methods and pretrained artifacts.

- SNASNet: https://github.com/Intelligent-Computing-Lab-Panda/Neural-Architecture-Search-for-Spiking-Neural-Networks
- AutoST: https://github.com/AlexandreWANG915/AutoST
- SpikingJelly: https://github.com/fangwei123456/spikingjelly

The SNASNet adapter follows the backward-cell matrix choices and delayed feedback of the source project. AutoST uses the choices in `experiments/search_space/Spikformer_space_tiny.yaml`, including per-block head counts and MLP ratios. Both are compact reference models, not copies of the published full training implementations. In particular, their feature extractors and model-level layouts are simplified; results from these adapters are not numerically interchangeable with scores of the original upstream architectures. No third-party checkpoints or datasets are bundled.
