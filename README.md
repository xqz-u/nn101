# nn101

## WIP

- [x] Categorical loss module (cross entropy).
- [x] Regression test: check that a single linear layer is able to learn a
      simple regression task.
- [x] SGD.
- [ ] Numerical gradient tests (finite differences).
  - [x] Linear layer
- [x] Fix static graph computation being broken by inconsistent batch size during training.

## TODO

- [ ] Refactor the `Tensor` struct to be more general, using a `shape` field
      instead of 3 hard-coded `[batch, n, features]` dimensions.
- [ ] Benchmark against implementation using strides.
- [ ] Benchmark against BLAS implementation version.
- [ ] ~~Benchmark dynamic vs. static computation graph allocation (?).~~ Does
      not seem super interesting...
- [ ] Document the whole process in some sort of essay.
- [ ] Compile with -O2 / -O3 + inspect assembly (-S) ?
- [ ] Logging ?
- [ ] Run tests on push/commit.
- [x] Configure `cloc`.
