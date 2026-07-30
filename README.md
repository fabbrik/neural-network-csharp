# Neural Network from Scratch in C#

A feed-forward neural network built from first principles in C# — no ML libraries, no
frameworks, just arrays and calculus. Backpropagation, SIMD-accelerated math, model
serialization, and a verified gradient check.

It comes with a [**study guide**](STUDY-GUIDE.md) that explains *why every piece exists*, from
"what is a neuron" through the chain rule to cache lines — including two worked examples you can
follow with a calculator.

```csharp
var net = new Sequential(inputs: 2)
    .Dense<Tanh>(4)
    .Dense<Sigmoid>(1)
    .Build(seed: 42);

net.Train(inputs, targets, epochs: 4000, learningRate: 0.5f);

float prediction = net.Predict([1f, 0f])[0];   // 0.9779

ModelIO.Save(net, "xor.nnm");
var loaded = ModelIO.Load("xor.nnm");          // predicts identically
```

## Why another one of these?

There are many "neural network from scratch" repos. Two things here are less common:

**It proves its own correctness.** A subtly wrong backward pass still trains *somewhat*, which
makes such bugs brutal to find. [`GradientCheck`](src/NN/GradientCheck.cs) compares every analytic
gradient against a central finite difference, and the test suite asserts the accuracy U-curve that
only a *correct* gradient produces — including a test with a deliberately broken derivative, to
prove the check can actually fail.

**It's honest about its limits.** [§24 of the study guide](STUDY-GUIDE.md) lists what this
implementation doesn't do and what it would cost, in order — starting with the admission that
batched GEMM would outweigh every SIMD optimization in the codebase.

## Running it

```bash
dotnet run --project src/NN.Demo    # perceptron on AND, network on XOR, save/load, gradient check
dotnet test                         # 42 tests
```

Output:

```
Perceptron on AND: converged in 4 epochs

Layer                     Output    Params
──────────────────────────────────────────
Dense<Tanh>                    4        12
Dense<Sigmoid>                 1         5
──────────────────────────────────────────
Input width: 2
Trainable parameters: 17

Network on XOR (2 -> 4 tanh -> 1 sigmoid):
  epoch  4000  loss 0.000350
  0 XOR 0 -> 0.0100    1 XOR 0 -> 0.9779
  0 XOR 1 -> 0.9826    1 XOR 1 -> 0.0226

Saved to xor.nnm (127 bytes), reloaded:
  round-trip is bit-for-bit identical

Gradient check: max relative error = 3.521E-004
```

## What's implemented

| | |
|---|---|
| **Layers** | Dense (fully connected), any depth |
| **Activations** | Sigmoid, Tanh, ReLU, Identity, Step |
| **Training** | Backpropagation, mini-batch SGD, MSE loss, shuffling |
| **Initialization** | Xavier/Glorot uniform |
| **Model API** | Keras-style `Sequential` builder, `Summary()` |
| **Persistence** | Versioned binary format with architecture + weights |
| **Verification** | Finite-difference gradient checking |

## Design notes

**Weights are stored unit-major in one flat array.** Unit `j`'s weights occupy
`Weights[j * Inputs .. (j+1) * Inputs]` — the transpose of NumPy's `(n, j)` layout. That makes
each dot product a contiguous SIMD walk instead of a strided gather, and it pays off again in the
backward pass, where both the weight-gradient accumulation and the input-gradient propagation
walk the same contiguous memory.

**Activations are generic type parameters, not delegates.** `Dense<Tanh>` uses C# 11 static
abstract interface members, so the JIT inlines the activation into the loop. A
`Func<float, float>` field would cost an un-inlinable indirect call per unit.

**SIMD adapts to the CPU.** [`SimdOps`](src/NN/SimdOps.cs) uses `Vector<float>`, which is 4 wide
on ARM NEON and 8 on AVX2, with two accumulators to keep the multiply-add pipeline fed and a
scalar tail for lengths that aren't a multiple of the width.

## Project layout

```
src/NN/           the library
src/NN.Demo/      runnable demonstration
tests/NN.Tests/   42 tests
STUDY-GUIDE.md    the long-form explanation
```

## Requirements

.NET 9 SDK. No dependencies beyond the framework; the test project uses xUnit.

## License

MIT — see [LICENSE](LICENSE).
