# CUDA Profiling Practice

This repository contains my ongoing practice while learning CUDA and getting more comfortable with compiling, running, and profiling GPU code on WAVE.

I’m using this repo as a working space to experiment with small examples, builds, and profiling workflows rather than as a finished project.

---

## Repository structure

### `lecture2/`
Cloned from the GPU Mode GitHub. Contains lecture code and examples used for CUDA and GPU programming practice.

### `lectures/`
Also from the GPU Mode GitHub. Includes additional lecture material and reference code.

### `profiling-cuda-in-torch/`
Material from GPU Mode focused on profiling CUDA code in PyTorch. Used to understand how CUDA kernels behave inside higher-level frameworks.

### `UdemyCuda/`
Small CUDA examples I’m working through to learn the basics.
- `helloWorld.cu` – minimal CUDA kernel example
- `hello` – compiled binary

### `build/`
CMake build outputs generated during configuration and compilation. These are kept to help debug build issues and understand how CMake sets up CUDA projects.

### `.vscode/`
VS Code configuration for CUDA development on the cluster (include paths, compiler settings, etc.).

---

## Purpose

This repo is mainly for:
- learning CUDA programming
- understanding nvcc and CMake workflows
- experimenting with profiling tools
- working through GPU Mode materials on real hardware

This is a work-in-progress learning repo, not a polished library.

---

## Status

Work in progress.
