# CUDA Profiling Practice

This repository contains my ongoing practice while learning CUDA and getting more comfortable with compiling, running, and profiling GPU code on WAVE.

I’m using this repo as a workspace to test small examples, understand build tools, and see how CUDA programs behave in a real HPC environment.

---

## Repository structure

### `UdemyCuda/`
Simple CUDA examples I’m working through to learn the basics.
- `helloWorld.cu` – minimal CUDA kernel example
- `hello` – compiled binary

### `lectures/` and `lecture2/`
Lecture code and supporting material from coursework and self-study. These are tracked as separate Git repositories since they come from different sources.

### `build/`
CMake build outputs generated during configuration and compilation. These files are kept to help debug build issues and understand how CMake sets up CUDA projects.

### `.vscode/`
VS Code configuration used for CUDA development on the cluster (include paths, compiler settings, etc.).

---

## Purpose

This repo is mainly for:
- learning CUDA programming
- understanding nvcc and CMake workflows
- experimenting with profiling and debugging on HPC systems

It’s not meant to be a polished project, just a place to keep track of what I’m learning.

---

## Status

Work in progress.
