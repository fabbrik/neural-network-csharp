# Benchmarks

The repo makes several performance claims. This project measures them, including the one that
turned out to be **wrong**.

```bash
dotnet run -c Release --project bench/NN.Bench -- --filter '*'
dotnet run -c Release --project bench/NN.Bench -- --filter '*DotProduct*'   # one group
```

Release configuration is required — BenchmarkDotNet refuses to run otherwise, and a Debug build
would measure nothing meaningful.

## The machine these numbers came from

> Apple M3 Pro (11 cores), macOS 26.3, **.NET 10.0.10**, Arm64 RyuJIT armv8.0-a, BenchmarkDotNet
> 0.15.8, `--job short`.

**`Vector<float>` is 4 wide here.** On an AVX2 x86 machine it is 8 and on AVX-512 it is 16, so the
SIMD ratios below will differ — the *shape* of every result should not. Re-run it on your own
hardware; that is the point of shipping the project rather than just the table.

## 1. SIMD, and the second accumulator

`SimdOps.Dot` against the obvious scalar loop, and against a vectorized version using a single
accumulator.

| Length | Scalar | 1 accumulator | 2 accumulators (shipping) | SIMD speedup | 2nd accumulator |
|---|---|---|---|---|---|
| 8 | 4.25 ns | 1.27 ns | 1.54 ns | **2.8×** | **0.83× — actively worse** |
| 64 | 37.9 ns | 9.87 ns | 8.02 ns | **4.7×** | **1.23×** |
| 512 | 357 ns | 78.8 ns | 59.3 ns | **6.0×** | **1.33×** |
| 4096 | 2975 ns | 713 ns | 481 ns | **6.2×** | **1.48×** |

**Both claims hold at realistic lengths.** Vectorization is worth 4.7–6.2× once there is enough
work to amortize the loop setup, and the second accumulator is worth a further ~1.2–1.5× — more
than the 4-wide vector alone would suggest, which is the pipelining effect the code comment
describes.

**At length 8 the second accumulator now costs 17%.** Under .NET 9 this row was a tie; on .NET 10
the single-accumulator version pulled ahead. Eight floats is exactly two 4-wide vectors, so the
two-accumulator loop runs its main body once and falls straight into the tail — all setup, no
pipelining to win back. The shipping code keeps two accumulators anyway, because the lengths that
matter are layer widths and the crossover sits well below any useful layer. But it is a fair
reminder that **"strictly better" optimizations usually aren't**, and that the sign of an effect
can flip on a runtime upgrade even when nothing in your source changed.

## 2. Weight layout: unit-major vs. feature-major

Identical weights, identical arithmetic, identical activation. The only difference is memory
order: unit-major (contiguous, what ships) vs. NumPy's feature-major `(inputs, units)` (strided).

| Layer shape | Unit-major | Feature-major | Cost of striding |
|---|---|---|---|
| 2 × 4 (the XOR layer) | 16.0 ns | 13.9 ns | **0.87× — striding is *faster*** |
| 64 × 64 | 727 ns | 3306 ns | **4.6×** |
| 784 × 128 (MNIST-sized) | 12.8 µs | 75.9 µs | **5.9×** |

**The claim holds, with a caveat the docs previously omitted.** At realistic sizes the contiguous
layout is worth 4.6–5.9×, which is the largest single effect measured here and justifies calling
it the most consequential design decision.

But at 2×4 it loses. Eight weights fit inside one 64-byte cache line, so there is no cache line to
waste and no gather to avoid — only the SIMD path's extra setup, which the scalar strided loop
skips. **The XOR demo is precisely the size at which none of this matters.** That is worth stating
plainly, because it is the example the reader meets first.

## 3. Generic activation vs. delegate — the claim that was wrong

The README used to assert that a `Func<float, float>` field "would cost an un-inlinable indirect
call per unit." It is un-inlinable. It costs almost nothing.

| Layer shape | `Dense<Tanh>` | Delegate (tanh) | `Dense<ReLU>` | Delegate (ReLU) |
|---|---|---|---|---|
| 2 × 4 | 16.0 ns | 17.9 ns (**1.12×**) | 9.86 ns | 9.77 ns (**0.99×**) |
| 64 × 64 | 727 ns | 737 ns (**1.01×**) | 594 ns | 602 ns (**1.01×**) |
| 784 × 128 | 12.79 µs | 12.68 µs (**0.99×**) | 12.65 µs | 12.85 µs (**1.02×**) |

**Measured cost: within ±2% at every realistic size, and the sign is not even consistent.** The
one outlier — 1.12× on the 2×4 tanh layer — is four units' worth of work at ~16 ns total, where a
couple of nanoseconds of call overhead is still a visible fraction. It is also the layer size at
which nothing about performance matters.

The first suspicion was that tanh — a transcendental costing tens of cycles — was hiding the call,
so the table repeats the comparison with ReLU, which is a compare and a select. The result barely
moves. The reason is arithmetic, not dispatch: the activation runs **once per unit**, while the
dot product feeding it runs `Inputs` multiply-adds per unit. At 784 inputs the indirect call is
amortized over 784 fused multiply-adds. It is invisible because it is rare, not because it is fast.

The generic design is still the better default — it composes with `readonly struct` activations at
zero cost and keeps the JIT free to inline — but it should be justified as a *type-safety and
composition* win, not a performance one. The docs now say so.

## 4. `ForwardBatch` — a deliberate null result

§25 of the study guide claims `ForwardBatch` buys nothing today because it just loops the
single-example path. Testing your own negative claims is the only way they stay true.

| Batch size | One at a time | `ForwardBatch` | Ratio |
|---|---|---|---|
| 1 | 2.55 µs | 2.62 µs | 1.03× |
| 32 | 91.8 µs | 93.4 µs | 1.02× |
| 256 | 802 µs | 754 µs | 0.95× (noisy — see below) |

**Confirmed: no benefit.** Every row is within a few percent of parity, and the direction flips
between rows — under .NET 9 all three read 0.98×, here two read slightly *slower*. The 256-row
baseline also carried a ±80 µs standard error against a 802 µs mean, so its 0.95× is measurement
noise rather than a win. Nothing here is a real effect in either direction, which is exactly the
claim.

A real tiled GEMM — reusing each loaded weight block across many examples instead of re-streaming
the whole matrix per example — is where the batching win lives, and it remains unimplemented. See
study guide §25 item 1 and exercise 14.

## What is not measured here

- **Training throughput** end to end (forward + backward + update), as opposed to a single forward pass.
- **The backward pass**, whose `AddScaled` is as hot as `Dot` and is not benchmarked separately.
- **Explicit FMA** vs. whatever `acc += a * b` compiles to (§25 item 5).
- **Anything multi-threaded** — the library is single-threaded throughout.

CI builds this project on every push so it cannot rot, but does not run it: benchmark timings from
shared cloud runners are not worth the minutes they cost.
