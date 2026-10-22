# Resource model

The resource graph uses four identities that must not be conflated.

## Node

A physical or virtual computer running an enrolled agent. A node may expose zero, one
or many compute devices. Roles such as `front_door`, `control`, `coordinator`, `compute`
and `cache` are policy capabilities, not guaranteed active roles.

## Device

A compute target exposed by a worker backend, for example a CUDA GPU, HIP integrated
GPU or CPU backend. Devices are owned by exactly one node.

## Physical memory pool

A capacity/pressure domain. Several devices may reference the same pool. This is
critical for unified-memory systems: CPU-visible RAM and iGPU-visible memory are not
summed simply because two devices can access them.

A pool can have a reported physical capacity, but admission uses an owner policy and
runtime observations. Installed capacity is not automatically allocatable capacity.

## Resource policy

The local owner's maximum grant for a pool. The global controller may choose a lower
budget but never a higher one. Safety headroom and allocation caps are represented
separately because they answer different questions.

## Capability evidence

Device identity alone is insufficient. Future capability records bind an execution
claim to OS, driver/runtime, worker build, model architecture, encoding, operators and
relevant transport behavior. Evidence is invalidated when those inputs change.
