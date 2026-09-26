# Third-party references

This implementation uses the published search-space definitions and documented interfaces of the following projects. Their repositories remain the authoritative sources for the original methods and pretrained artifacts.

- SNASNet: https://github.com/Intelligent-Computing-Lab-Panda/Neural-Architecture-Search-for-Spiking-Neural-Networks
- AutoST: https://github.com/AlexandreWANG915/AutoST
- SpikingJelly: https://github.com/fangwei123456/spikingjelly

The local state-explicit model adapters follow the architecture variables of SNASNet and AutoST. They are compact reference models, not copies of the published full training implementations. No third-party checkpoints or datasets are bundled.
