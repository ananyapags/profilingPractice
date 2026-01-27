# CUDA Profiling Practice

This repository contains my ongoing work while learning CUDA and getting more comfortable with compiling, running, and profiling GPU code on WAVE.

I’m using this repo as a working space to experiment with small CUDA examples, build systems, and profiling workflows. It’s meant to track what I’m learning rather than serve as a finished or polished project.

---

## Repository structure

### `lecture2/`
Cloned from the GPU Mode GitHub. Contains lecture code and examples used for CUDA and GPU programming practice.

### `lectures/`
Additional lecture material from the GPU Mode GitHub, used alongside coursework and self-study.

### `profiling-cuda-in-torch/`
GPU Mode material focused on profiling CUDA kernels in PyTorch, mainly to understand how low-level GPU behavior shows up in higher-level frameworks.

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

This is a work-in-progress learning repository.

---

## Status

In progress.
