# Benchmarks

***English** · [Español](README.es.md)*

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
> 0.15.8. Every section uses BenchmarkDotNet's default job — see the note under section 1 for why
> `--job short` is not good enough to draw a conclusion from.

**`Vector<float>` is 4 wide here.** On an AVX2 x86 machine it is 8 and on AVX-512 it is 16, so the
SIMD ratios below will differ — the *shape* of every result should not. Re-run it on your own
hardware; that is the point of shipping the project rather than just the table.

## 1. SIMD, and the second accumulator

`SimdOps.Dot` against the obvious scalar loop, and against a vectorized version using a single
accumulator.

| Length | Scalar | 1 accumulator | 2 accumulators (shipping) | SIMD speedup | 2nd accumulator |
|---|---|---|---|---|---|
| 8 | 4.23 ns | 1.27 ns | 1.03 ns | **4.1×** | **1.23×** |
| 64 | 38.3 ns | 9.92 ns | 7.91 ns | **4.8×** | **1.25×** |
| 512 | 360 ns | 82.4 ns | 63.5 ns | **5.7×** | **1.30×** |
| 4096 | 2937 ns | 723 ns | 500 ns | **5.9×** | **1.45×** |

**Both claims hold.** Vectorization is worth 4.1–5.9× once there is enough work to amortize the
loop setup, and the second accumulator is worth a further ~1.2–1.5× — more than the 4-wide vector
alone would suggest, which is the pipelining effect the code comment describes.

> **These rows were measured with `--job short` in an earlier revision and read differently.**
> Length 8 in particular showed the second accumulator *losing* 17%, and the text here explained
> why at some length. On the default job it wins 1.23× instead. The short job's ±0.26 ns error bar
> was simply wider than the effect it was being used to describe. The lesson is worth keeping even
> though the finding did not survive: **a three-iteration job is for triage, not for conclusions.**

### Why not just call `TensorPrimitives.Dot`?

Because it loses here, which is not what you would guess. `System.Numerics.Tensors` ships
hand-tuned kernels, and the obvious move is to delete the loop above and call one.

| Length | 1 accumulator | 2 accumulators (shipping) | `TensorPrimitives.Dot` |
|---|---|---|---|
| 8 | 1.27 ns | **1.03 ns** | 2.35 ns (**2.3× slower**) |
| 64 | 9.92 ns | 7.91 ns | **5.98 ns** (1.32× faster) |
| 512 | 82.4 ns | **63.5 ns** | 61.4 ns (tie) |
| 4096 | 723 ns | **500 ns** | 734 ns (**1.47× slower**) |

Look at the 4096 row: `TensorPrimitives.Dot` measures within 2% of the *single*-accumulator
version. A dot product ends in a reduction, and the kernel carries one accumulator chain through
it — the exact serial dependency the second accumulator exists to break. At length 8 it also loses
badly, because it is a real call where `SimdOps.Dot` is inlined, and the call costs more than the
arithmetic. So `Dot` stays hand-rolled.

This cuts the other way for `AddScaled` — see section 5 — which is why the library uses
`TensorPrimitives` for one of its two primitives and not the other. Neither choice was predictable
from the API surface; both came from this table.

## 2. Weight layout: unit-major vs. feature-major

Identical weights, identical arithmetic, identical activation. The only difference is memory
order: unit-major (contiguous, what ships) vs. NumPy's feature-major `(inputs, units)` (strided).

| Layer shape | Unit-major | Feature-major | Cost of striding |
|---|---|---|---|
| 2 × 4 (the XOR layer) | 19.1 ns | 16.0 ns | **0.84× — striding is *faster*** |
| 64 × 64 | 525 ns | 3249 ns | **6.2×** |
| 784 × 128 (MNIST-sized) | 9.75 µs | 72.6 µs | **7.4×** |

**The claim holds, with a caveat the docs previously omitted.** At realistic sizes the contiguous
layout is worth 6.2–7.4×, which is the largest single effect measured here and justifies calling
it the most consequential design decision.

But at 2×4 it loses. Eight weights fit inside one 64-byte cache line, so there is no cache line to
waste and no gather to avoid — only the SIMD path's extra setup, which the scalar strided loop
skips. **The XOR demo is precisely the size at which none of this matters.** That is worth stating
plainly, because it is the example the reader meets first.

## 3. Generic activation vs. delegate — the claim that was wrong

The README used to assert that a `Func<float, float>` field "would cost an un-inlinable indirect
call per unit." It is un-inlinable. It costs almost nothing.

Note the control column. `Dense<Tanh>` now activates a whole layer per call (section 6), while the
delegate reference still activates per unit, so those two differ by more than dispatch and cannot
answer the dispatch question. `ScalarActivation` is the honest control: identical to the delegate
version in every respect *except* that its activation is a generic type parameter instead of a
`Func` field.

| Layer shape | Generic, per-unit | Delegate, per-unit | Cost of dispatch | (shipping `Dense<Tanh>`) |
|---|---|---|---|---|
| 2 × 4 | 15.4 ns | 18.3 ns | **1.19×** | 19.1 ns |
| 64 × 64 | 752 ns | 741 ns | **0.99×** | 525 ns |
| 784 × 128 | 12.71 µs | 13.16 µs | **1.04×** | 9.75 µs |

**Measured cost: within ±4% at every realistic size, and the sign is not even consistent.** The
one outlier — 1.19× on the 2×4 tanh layer — is four units' worth of work at ~15 ns total, where a
couple of nanoseconds of call overhead is still a visible fraction. It is also the layer size at
which nothing about performance matters.

The first suspicion was that tanh — a transcendental costing tens of cycles — was hiding the call,
so the benchmark repeats the comparison with ReLU, which is a compare and a select. The result
barely moves. The reason is arithmetic, not dispatch: the activation runs **once per unit**, while
the dot product feeding it runs `Inputs` multiply-adds per unit. At 784 inputs the indirect call is
amortized over 784 multiply-adds. It is invisible because it is rare, not because it is fast.

The generic design is still the better default — it composes with `readonly struct` activations at
zero cost and keeps the JIT free to inline — but it should be justified as a *type-safety and
composition* win, not a performance one. The docs now say so.

## 4. `ForwardBatch` — a deliberate null result

§25 of the study guide claims `ForwardBatch` buys nothing today because it just loops the
single-example path. Testing your own negative claims is the only way they stay true.

| Batch size | One at a time | `ForwardBatch` | Ratio |
|---|---|---|---|
| 1 | 1.87 µs | 1.86 µs | 0.99× |
| 32 | 57.9 µs | 58.1 µs | 1.00× |
| 256 | 481 µs | 469 µs | 0.97× |

**Confirmed: no benefit.** Every row is within 3% of parity, well inside the run-to-run spread.
Nothing here is a real effect in either direction, which is exactly the claim.

(These rows are 1.4–1.6× faster in absolute terms than the previous revision measured, for the
reason in section 6. Both columns moved together, so the ratio — the only thing this section
claims — is unchanged.)

A real tiled GEMM — reusing each loaded weight block across many examples instead of re-streaming
the whole matrix per example — is where the batching win lives, and it remains unimplemented. See
study guide §25 item 1 and exercise 14.

## 5. `AddScaled` — where `TensorPrimitives` does win

`dest += src * scale`, the backward pass's workhorse: it runs twice per unit, accumulating weight
gradients and propagating the input gradient, so it does more of the training work than `Dot` does.

| Length | Hand-rolled `Vector<float>` | `TensorPrimitives.MultiplyAdd` | Ratio |
|---|---|---|---|
| 8 | **1.29 ns** | 2.84 ns | **2.19× slower** |
| 64 | 9.87 ns | **8.66 ns** | 1.14× faster |
| 512 | 74.6 ns | **29.6 ns** | **2.52× faster** |
| 4096 | 566 ns | **222 ns** | **2.55× faster** |

**2.5× at any length worth vectorizing, so this one ships.** The opposite outcome to `Dot`, from
the same library, for a structural reason: this is a pure streaming operation with no reduction,
so there is no serial dependency chain to carry and nothing stopping the kernel from unrolling as
wide as it likes. It also emits a real fused multiply-add where `dest[i] += src[i] * scale`
compiles to a separate multiply and add.

The length-8 row loses for the same reason it does in `Dot` — an un-inlined call around three
vector instructions — and for the same reason it does not matter: the lengths that reach here are
layer widths.

## 6. Activating a layer at a time instead of a unit at a time

`exp` and `tanh` cost tens of cycles per call, and a layer calls one per unit. Applying the
activation to the whole output vector at once, after the dot products rather than inside them,
lets `TensorPrimitives` do four at a time. Measured on the activation alone, with the dot products
out of the picture:

| Width | Sigmoid scalar → vector | Tanh scalar → vector | Softmax scalar → vector |
|---|---|---|---|
| 10 | 16.6 → 16.0 ns (1.04×) | 19.9 → 18.2 ns (1.09×) | 24.3 → 34.6 ns (**0.70×**) |
| 128 | 208 → 92.8 ns (**2.24×**) | 242 → 121 ns (**2.00×**) | 289 → 134 ns (**2.15×**) |
| 1024 | 1643 → 749 ns (**2.21×**) | 1914 → 956 ns (**2.00×**) | 2219 → 1070 ns (**2.07×**) |

**Sigmoid and tanh ship vectorized; softmax does not.** The width column is why. A hidden layer is
as wide as you make it, so 128 and 1024 are the normal cases and 2× is real. But a softmax layer
is *one unit per class* — ten, for MNIST — and at ten it loses 1.4×, because the numerically
stable form needs a max-subtraction pass before the kernel and four short vectorized passes beat
three short scalar ones only once the passes stop being short. It only pulls ahead past roughly a
hundred classes. `SoftmaxCrossEntropy.Transform` therefore stays a scalar loop.

`TensorPrimitives.SoftMax` cannot be used directly in any case: it computes `exp(z)/Σexp(z)`
literally, with no max-subtraction, so it returns `NaN` for logits above ~88. Two existing unit
tests caught that immediately.

### The 6× regression this change caused, and the one-word fix

Folding the two changes into `Dense.Forward` — compute every pre-activation, then activate the
vector — made the 784×128 layer **6.3× slower**, 12.79 µs to 80 µs. The arithmetic was identical
and the same code measured 10.7 µs when called from a benchmark directly.

The disassembly showed why. Inlined into `Dense<TActivation>.Forward` — generic, and by then also
holding an inlined `Dot` and an inlined vectorized activation — the JIT stopped eliminating bounds
checks on the inner vector loads, leaving a range-check branch guarding every `ldr q`:

```asm
ldr     q18, [x22]
cmp     x21, x11
bhi     G_M000_IG35      ← per vector, per iteration
```

The fix is `[MethodImpl(MethodImplOptions.NoInlining)]` on `SimdOps.MatVec`. Compiled on its own —
small, non-generic — it optimizes cleanly, and the layer pays one ordinary call. The final layer
is **9.75 µs, 1.31× faster than before any of this**, and `Dense<ReLU>` gained the same 1.29×
without any change to its activation at all: the original fused loop had been losing that much to
the same inlining pessimization all along.

Worth stating plainly, because it is the most useful thing in this file: **a local optimization
made the thing it optimized six times slower, and only the benchmark said so.** Nothing about the
source suggested it. Re-measure `LayerBenchmarks` before touching `MatVec`.

## 7. Does any of it show up in training?

Sections 1–6 measure primitives and forward passes. Training is what the library actually spends
its time doing, and a primitive that wins 2.5× in isolation has not earned anything until it is
weighed against everything it shares a step with. One mini-batch — forward, backward, and the
descent update — on `784 → 128 tanh → 10 softmax`, batch size 32:

| | With `TensorPrimitives` | Hand-rolled `AddScaled` | |
|---|---|---|---|
| **Full step** | **607.5 µs** | 815.0 µs | **1.34× faster** |
| Forward only *(control)* | 318.9 µs | 317.9 µs | 1.00× — unchanged |
| **Backward half** *(by subtraction)* | **288.6 µs** | 497.1 µs | **1.72× faster** |

**The package earns its place: a third off every training step.** The control row is what makes
that trustworthy. `AddScaled` appears nowhere in the forward pass, so if the two forward numbers
had diverged, something other than the intended change would have moved and the whole comparison
would be suspect. They differ by 0.3%, inside the noise, which places the entire 207 µs difference
in the backward pass — exactly where the primitive lives.

**Backprop is where this library spends its time, and section 5 undersold it.** The backward half
is 48% of the step here, and it runs `AddScaled` twice per unit against `Dot`'s once, so the
primitive is a larger share of training than any forward-pass benchmark can show. That is why a
2.5× on a microbenchmark became 1.72× on the pass and 1.34× on the whole step, rather than the
rounding error a forward-only view would have predicted.

It also settles a question sections 1–6 could not: dropping the dependency to keep the library
package-free would cost a third of its training throughput.

## What is not measured here

- **Anything multi-threaded** — the library is single-threaded throughout.
- **Batched GEMM**, the one remaining large win — see section 4 and study guide §25.
- **Whole-epoch or whole-run timing**, including data loading and shuffling. Section 7 measures one
  mini-batch in isolation; the MNIST demo's end-to-end 37 s is reported in the README, not here.

CI builds this project on every push so it cannot rot, but does not run it: benchmark timings from
shared cloud runners are not worth the minutes they cost.
