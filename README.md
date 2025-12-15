# nn101

## WIP

- [x] First working MLP implementation.
  - [ ] Better design!
- [x] Categorical loss module (cross entropy).
- [x] Regression test: check that a single linear layer is able to learn a
      simple regression task.
- [x] SGD.
- [ ] Numerical gradient tests (finite differences).
  - [x] Linear layer
- [x] Fix static graph computation being broken by inconsistent batch size during training.

## TODO

- [ ] `Tensor`:
  - [ ] Refactor the struct to be more general, using a `shape` field instead of
        3 hard-coded `[batch, n, features]` dimensions.
  - [ ] Differentiate between `flat_size` (product of dimensions) & `capacity`
        (occupied physical memory), write a `reshape()` operation that when
        changing dimensions also takes care of these derived properties.
- [ ] Benchmarks:
  - [ ] Implement strides & use them for things like reshape and transpose.
  - [ ] Use linear algebra BLAS primitives.
  - [ ] Parallel version.
  - [ ] CUDA ?
- [ ] Profiling
- [ ] Document the whole process in some sort of essay.
- [ ] Compile with -O2 / -O3 + inspect assembly (-S) ?
- [ ] Logging ?
- [ ] Run tests on push/commit.
- [x] ~~Explore the concept of [handles](https://stackoverflow.com/a/21501054) to~~
      ~~define structures in source files rather than headers - I'm working mostly~~
      ~~with struct pointers anyways.~~ **NOTE** No auto-completion, inconvenient...
- [x] ~~Benchmark dynamic vs. static computation graph allocation (?).~~ Does
      not seem super interesting...
- [x] Configure `cloc`.
