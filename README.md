# Kernel Tutorials

Interactive, presentation-style tutorials on [tt-metal](https://github.com/tenstorrent/tt-metal)
kernel programming, built as companions to the `kernel_dojo` exercises.

**Live site:** https://mstojkovicTT.github.io/kernel-tutorials/

| Tutorial | What it covers |
|---|---|
| [Circular Buffer Dojo](https://mstojkovicTT.github.io/kernel-tutorials/circular-buffers/) | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, the DST handshake, blocking & batching, and the classic deadlock/wrong-data bugs — as live simulations of the dojo exercise kernels |

Each tutorial is a single self-contained HTML file (no build step, no dependencies).
Navigate slides with ← / → ; on simulation slides, **step** advances every simulated
RISC-V processor by one instruction.

## Adding a tutorial

1. Drop a self-contained `index.html` into a new directory, e.g. `sharding/index.html`.
2. Add a card for it in the root `index.html`.
3. Push — GitHub Pages serves it at `/sharding/`.
