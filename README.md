# Neural Network from Scratch in C#

A feed-forward neural network built from first principles in C# — no ML libraries, no
frameworks, just arrays and calculus. Backpropagation, SIMD-accelerated math, model
serialization, a verified gradient check, and benchmarks for every performance claim it makes.

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

There are many "neural network from scratch" repos. Three things here are less common:

**It verifies its own backward passes.** A subtly wrong backward pass still trains *somewhat*,
which makes such bugs brutal to find. [`GradientCheck`](src/NN/GradientCheck.cs) compares every
analytic gradient against a central finite difference, and the test suite asserts the accuracy
U-curve that only a *correct* implemented gradient produces — including a test with a deliberately
broken derivative, to prove the check can actually fail.

**It measures its own claims, and reports the one that was wrong.** Every performance assertion in
these docs has a [benchmark](bench/) behind it. One of them — that a delegate-dispatched activation
would be meaningfully slower than a generic one — turned out to be **false**, and
[the write-up says so](bench/README.md#3-generic-activation-vs-delegate--the-claim-that-was-wrong)
rather than quietly dropping it.

**It's honest about its limits.** [§25 of the study guide](STUDY-GUIDE.md) lists what this
implementation doesn't do and what it would cost, in order — starting with the admission that
batched GEMM would outweigh every SIMD optimization in the codebase, a claim the benchmarks
now [partially confirm](bench/README.md#4-forwardbatch--a-deliberate-null-result).

## Running it

```bash
dotnet run --project src/NN.Demo      # perceptron, XOR, save/load, gradient check, two moons
dotnet run -c Release --project src/NN.Mnist   # handwritten digit recognition (~40 s)
dotnet test                           # the full suite
dotnet run -c Release --project bench/NN.Bench -- --filter '*'   # benchmarks
```

The demo runs six sections. The first four are XOR-scale — the perceptron converging on AND, a
hidden layer solving XOR, a model round-tripping through disk, and a gradient check:

```
Perceptron on AND: converged in 4 epochs

Network on XOR (2 -> 4 tanh -> 1 sigmoid):
  epoch  4000  loss 0.000350
  0 XOR 0 -> 0.0100    1 XOR 0 -> 0.9779
  0 XOR 1 -> 0.9826    1 XOR 1 -> 0.0226

Saved to xor.nnm (127 bytes), reloaded:
  round-trip is bit-for-bit identical

Gradient check: max relative error = 3.521E-004
```

The last two use a dataset big enough to show what XOR structurally cannot — mini-batching,
generalization to held-out data, and overfitting:

```
Two moons: 1500 noisy points, 1000 train / 500 test

Training (batch size 32, learning rate 0.3):
  epoch   train loss   train acc   test acc
      1       0.1511      83.9%     85.2%
    150       0.0177      98.1%     96.4%

Learned decision boundary (· = class 0, # = class 1):

  #####################################################·······
  ##################################################··········
  ###############################################·············
  #####################······###################··············
  ###################··········###############················
  #################··············############·················
  ################··················#######···················
  ##############··············································
  ############················································
  #########···················································

Overfitting: same problem, 20 training points, a much larger network
  4417 parameters, 20 training examples — 221x more knobs than data points

  epoch    train acc   test acc
   3000      100.0%     93.4%
```

> **On the exact digits.** Float addition is not associative, and `Vector<float>` is 4 wide on ARM
> but 8 on AVX2 — so the SIMD dot product sums in a different order on different CPUs. Numbers
> quoted here and in the study guide come from an Apple M3 Pro on .NET 10. Expect the last digits to
> move on other hardware; expect the conclusions not to. CI deliberately runs on both
> architectures.

## Reading handwritten digits

A separate demo trains on [MNIST](src/NN.Mnist/) — 60,000 handwritten digits, 784 inputs, 101,770
parameters. **97.4% on digits it has never seen, in 37 seconds**, from two dense layers and
backpropagation:

```
A training example (label 5):
              ==++**..**@@@@==
    ....--****@@@@@@@@@@%%**@@@@##::
  ..@@@@@@@@@@@@@@@@@@@@------::..
    %%@@@@@@@@@@####@@@@
          **@@--
            ##@@::
              --@@@@@@==
                    @@@@@@::
            ..++%%@@@@@@@@##
      ::%%@@@@@@@@##--
  **%%@@@@@@@@##--

  epoch   train loss   test acc      elapsed
      1      0.02343    91.87%        2.1s
     10      0.00573    96.55%       18.3s
     20      0.00352    97.41%       36.4s

Confusion matrix — rows are the true digit, columns the prediction:

             0     1     2     3     4     5     6     7     8     9    accuracy
    0      969     ·     ·     1     ·     3     3     1     2     1      98.9%
    1        ·  1123     3     1     ·     1     3     1     3     ·      98.9%
    4        1     ·     1     1   956     ·     6     3     2    12      97.4%
    7        ·    10    14     2     ·     ·     ·   989     2    11      96.2%
```

It then prints the digits it got **wrong**, which is the more honest picture of "97%" than the
number is — most are ambiguous enough that a person would hesitate too.

**The trained model is saved, so you only pay for training once.** Re-running loads 397 KB of
weights instead of retraining: **37 s → 4 ms**, same 97.41%.

```bash
dotnet run -c Release --project src/NN.Mnist                  # reuse the saved model
dotnet run -c Release --project src/NN.Mnist -- --predict 42  # classify one image
dotnet run -c Release --project src/NN.Mnist -- --retrain     # train from scratch
```

`--predict` shows all ten outputs, so you can see what it nearly said:

```
  This is a 4. The network says 4 — correct.

    3  0.001
    4  0.974  ██████████████████████████████████████
    7  0.005
    9  0.017
```

That runner-up is the 4/9 confusion from the matrix above, visible in a single prediction.

After saving, the demo reloads the file and checks that 1,000 predictions come back **bit-for-bit
identical** — a serializer that drops or reorders parameters yields a model that still loads,
still predicts, and is merely *worse*, which is the same silent-failure class as a wrong gradient.

The dataset is not in this repository. The demo downloads it once (~11 MB) and caches it outside
the working tree; every later run, including offline ones, reads the cache. With no network and no
cache it explains that and exits cleanly rather than failing — so `dotnet test` and the other demo
never depend on a dataset mirror being up.

That 97.4% is roughly par for the architecture, and the gap to the ~98% usually quoted is
*explained by the library's documented limits, not by mystery*: MSE instead of softmax +
cross-entropy, and plain SGD with no momentum. The demo needs a learning rate of **1.0** — far
above the 0.1–0.5 the study guide recommends — and that is the MSE-over-sigmoid gradient being
visibly weak. Study-guide exercises 10 and 11 are those two fixes, with 97.4% in 37 s as the
number to beat.

## What's implemented

| | |
|---|---|
| **Layers** | Dense (fully connected), any depth |
| **Activations** | Sigmoid, Tanh, ReLU, Identity, Step |
| **Training** | Backpropagation, mini-batch SGD, MSE loss, shuffling |
| **Initialization** | Xavier/Glorot uniform |
| **Model API** | Keras-style `Sequential` builder, `Summary()` |
| **Persistence** | Versioned binary format with architecture + weights; train once, reload in ms |
| **Verification** | Finite-difference gradient checking |
| **Data** | Generated two-moons dataset; MNIST loader (IDX format, download + cache) |

## Design notes

Each of these was measured; the numbers link to [`bench/`](bench/README.md).

**Weights are stored unit-major in one flat array.** Unit `j`'s weights occupy
`Weights[j * Inputs .. (j+1) * Inputs]` — the transpose of NumPy's `(n, j)` layout. That makes
each dot product a contiguous SIMD walk instead of a strided gather, and it pays off again in the
backward pass, where both the weight-gradient accumulation and the input-gradient propagation
walk the same contiguous memory. **Measured: 4.6–5.9× on 64×64 and 784×128 layers** — the largest
effect in the codebase. It is worth *nothing* on the 2×4 XOR layer, whose eight weights fit inside
a single cache line; there, the strided version is marginally faster.

**SIMD adapts to the CPU.** [`SimdOps`](src/NN/SimdOps.cs) uses `Vector<float>`, which is 4 wide
on ARM NEON and 8 on AVX2, with two accumulators to keep the multiply-add pipeline fed and a
scalar tail for lengths that aren't a multiple of the width. **Measured: 4.7–6.2× over a scalar
loop, and the second accumulator is worth a further 1.2–1.5×** at any length longer than a couple
of vectors — below that it costs a little, which the benchmark write-up doesn't hide.

**Activations are generic type parameters, not delegates.** `Dense<Tanh>` uses C# 11 static
abstract interface members, so the JIT inlines the activation into the loop. This README used to
claim that the obvious alternative — a `Func<float, float>` field — would cost an un-inlinable
indirect call per unit and therefore be slower. **It is un-inlinable, and it is not slower:
measured within ±2% at any realistic size, with the sign not even consistent.** The activation
runs once per *unit* while the
dot product feeding it runs `Inputs` multiply-adds, so the call is amortized into invisibility.
The generic design stays, on its real merits — zero-cost composition with `readonly struct`
activations and no delegate allocation — but not for speed.

**Forward-for-inference and forward-for-training are separate methods.** `Forward` computes
activations; `ForwardTrain` also caches what the backward pass needs. When one method did both,
any incidental forward pass — evaluating a loss, logging a prediction mid-epoch — silently
overwrote the cache, so the next `Backward` differentiated the wrong example without erroring.
The split makes that unrepresentable, and `Backward` throws if no `ForwardTrain` preceded it.

## Threading and buffer ownership

The library is **single-threaded by design and lends out its buffers**. Two rules:

- **One network per thread.** `Network` and `Dense` hold mutable activation buffers, gradient
  accumulators, and shuffle state. Nothing is synchronized. (`Dense.Forward` on a standalone
  layer touches no instance state; `Network.Predict` still writes shared activation buffers.)
- **`Predict` returns a view the next call overwrites.** Copy it with `.ToArray()` if you need to
  keep it, and never hold two prediction results at once.

`ModelIO.Register` mutates a process-wide table. The table access is synchronized, but registering
custom layer types during start-up is still clearer than letting load behavior depend on timing.

## Project layout

```
src/NN/           the library
src/NN.Demo/      runnable demonstration — perceptron, XOR, two moons
src/NN.Mnist/     handwritten digit recognition, with an IDX reader and dataset cache
tests/NN.Tests/   the test suite
bench/NN.Bench/   benchmarks, with results in bench/README.md
STUDY-GUIDE.md    the long-form explanation
```

## Requirements

.NET 10 SDK. [`global.json`](global.json) pins the major version, so a machine with several SDKs
installed builds this repo with the one it was tested against.

The library has no dependencies beyond the framework; the test project uses xUnit and the
benchmarks use BenchmarkDotNet.

## License

MIT — see [LICENSE](LICENSE).
