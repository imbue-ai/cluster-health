# Training Infrastructure Scripts

This repository contains various scripts developed at Imbue for managing a large cluster of H100s, detecting and fixing hardware issues, and generally ensuring smooth model training. You can read more about our process [here](https://imbue.com/research/70b-infrastructure/)

The code is organized as follows:
- `gpu_stress_test` tests that the GPUs on each machine are able to allocate large tensors and perform standard operations/
- `health_checks` contains various checks we use to determine which hosts are healthy.
- `host_validation` contains tests to check that the GPUs on a given machine are able to communicate with each other (via NVLink) and with GPUs on other machines (via InfiniBand).
- `ufm_events` contains a script which parses the UFM event log, checks for relevant events, and determines which network ports should be disabled.
