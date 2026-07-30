# Neural Networks from Scratch in C# — Study Guide

A complete, working neural network with backpropagation, written in C# with SIMD — and an
explanation of **why every piece exists**, starting from zero assumed knowledge.

**How to use this guide.** Part I explains the ideas with no code — read it first, even if
you're impatient. Part II maps each idea onto the actual C# and explains the engineering.
Part III is practice, debugging, and exercises. The worked examples in §7 and §8 have been
computed and verified; follow along with a calculator and you will genuinely understand
backpropagation, which is the one concept everything else depends on.

| File | Contents |
|---|---|
| [`Activations.cs`](src/NN/Activations.cs) | `IActivation` and the activation functions |
| [`Perceptron.cs`](src/NN/Perceptron.cs) | The 1958 single-unit perceptron and its update rule |
| [`Dense.cs`](src/NN/Dense.cs) | The dense (fully connected) layer: forward, backward, gradient step |
| [`SimdOps.cs`](src/NN/SimdOps.cs) | Vectorized `Dot` and `AddScaled` |
| [`ILayer.cs`](src/NN/ILayer.cs) | Non-generic layer interface, so a network can mix activation types |
| [`Network.cs`](src/NN/Network.cs) | Layer stack, mini-batch SGD training loop, MSE loss, `Summary()` |
| [`Sequential.cs`](src/NN/Sequential.cs) | Keras-style fluent builder with input-size inference |
| [`ModelIO.cs`](src/NN/ModelIO.cs) | Saving and loading trained models |
| [`GradientCheck.cs`](src/NN/GradientCheck.cs) | Finite-difference verification of the backward pass |
| [`Program.cs`](src/NN.Demo/Program.cs) | Demos: perceptron on AND, network on XOR |

---

## Contents

**Part I — The Concepts** *(no code; read this first)*

1. [What problem is a neural network actually solving?](#1-what-problem-is-a-neural-network-actually-solving)
2. [The neuron](#2-the-neuron)
3. [Layers and the network](#3-layers-and-the-network)
4. [Learning = gradient descent](#4-learning--gradient-descent)
5. [Measuring wrongness: the loss function](#5-measuring-wrongness-the-loss-function)
6. [The chain rule](#6-the-chain-rule--the-one-piece-of-calculus-you-need)
7. [**Worked example: one neuron, by hand**](#7-worked-example-one-neuron-by-hand)
8. [**Worked example: the chain across two layers**](#8-worked-example-the-chain-across-two-layers)
9. [The perceptron, XOR, and why this history matters](#9-the-perceptron-xor-and-why-this-history-matters)

**Part II — The C# Implementation**

10. [Reading order](#10-reading-order)
11. [C# features you may not have met](#11-c-features-used-here-that-you-may-not-have-met)
12. [Data layout](#12-data-layout--the-most-consequential-decision)
13. [Activations in code](#13-activations-in-code)
14. [The forward pass in code](#14-the-forward-pass-in-code)
15. [SIMD](#15-simd--doing-several-multiplications-at-once)
16. [The backward pass in code](#16-the-backward-pass-in-code)
17. [Weight initialization](#17-weight-initialization--genuinely-not-optional)
18. [The perceptron in code](#18-the-perceptron-in-code)
19. [The network, Sequential, and saving models](#19-the-network-and-the-sequential-builder)
20. [The training loop](#20-the-training-loop)
21. [Gradient checking](#21-gradient-checking--how-you-know-the-code-is-right)

**Part III — Practice**

22. [Results](#22-results)
23. [Debugging playbook](#23-debugging-playbook)
24. [Exercises](#24-exercises)
25. [What this implementation does *not* do](#25-what-this-implementation-does-not-do)
26. [Where to go next](#26-where-to-go-next)
27. [Glossary](#glossary)

> **If you only read three sections:** §7 and §8 (backpropagation worked by hand) and §21
> (how you know it's right).

---
---

# Part I — The Concepts

---

## 1. What problem is a neural network actually solving?

You have examples of inputs paired with correct outputs:

| Input (x) | Correct output (y) |
|---|---|
| 0, 0 | 0 |
| 0, 1 | 1 |
| 1, 0 | 1 |
| 1, 1 | 0 |

You want a function that reproduces this mapping — and, more importantly, generalizes to
inputs it hasn't seen. You don't know the formula. So instead of writing one, you:

1. Build a function with **thousands of adjustable numbers** in it (the *parameters*).
2. Define a score for how wrong its outputs are (the *loss*).
3. Work out, for every parameter, which way to nudge it to reduce the loss.
4. Nudge them all a little in that direction, and repeat.

That's the whole idea. Two names get attached to the end of that list, and it's worth keeping
them straight from the start because they're constantly confused:

- **Backpropagation** is step 3 — *computing* the gradients efficiently. It's the clever part,
  and the thing this codebase exists to demonstrate.
- **Gradient descent** is step 4 — *applying* them.

Backprop tells you which way is downhill; gradient descent takes the step. Everything else in
this repo is bookkeeping and speed.

> The network isn't "reasoning." It's a large parameterized formula being fitted to data,
> the same way you'd fit a line to points — just with far more parameters and a nonlinear shape.

---

## 2. The neuron

A single neuron (here: a **unit**) does exactly three things.

**Step 1 — weighted sum.** Each input gets a weight saying how much it matters, positive or
negative. Add them up, plus a bias:

$$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

Think of it as a **vote**. Each input pushes the result up or down in proportion to its
weight. The bias `b` is the neuron's baseline — its opinion before seeing any input at all.
Without a bias, `z` would always be 0 when all inputs are 0, which is a needless restriction.

**Step 2 — activation.** Feed `z` through a nonlinear function `g`:

$$a = g(z)$$

**Step 3 — output `a`**, which becomes an input to the next layer.

### Why the activation is not optional

This is the most commonly skipped point, and it's important.

Suppose you skip the activation, so each layer is just a weighted sum. Stack two layers:
layer 2 computes a weighted sum of layer 1's outputs, which are themselves weighted sums of
the inputs. A weighted sum of weighted sums is... **still just a weighted sum**. Algebraically
you can collapse the two layers into one:

$$W_2(W_1x) = (W_2W_1)x = W_{\text{combined}}\,x$$

So a hundred stacked linear layers have exactly the power of a single linear layer — they can
only draw straight lines. The nonlinear `g` between layers is what stops the collapse and is
the entire reason depth buys you anything.

**Nonlinearity is what makes a "deep" network deep.** Remove it and depth is decoration.

---

## 3. Layers and the network

A **layer** is a group of units that all see the same inputs but have their own weights, so
each learns to detect something different.

```
     inputs            hidden layer          output layer
                      (4 units, tanh)       (1 unit, sigmoid)

                       ┌──► [u1] ──┐
                       ├──► [u2] ──┤
       x1 ─────────────┼──► [u3] ──┼──► [out] ──► prediction
                       ├──► [u4] ──┘
                       │      ▲
       x2 ─────────────┴──────┘

    x1 and x2 EACH connect to ALL FOUR units — 8 weights in the hidden layer.
    (Drawn merged to save space; there are no shared connections.)
```

That is the exact XOR network in [`Program.cs`](src/NN.Demo/Program.cs).

- **Dense / fully connected** means every input connects to every unit. A layer with `n`
  inputs and `j` units therefore has `n × j` weights plus `j` biases.
- The **hidden layer** is "hidden" only in that its outputs are never observed directly —
  they're intermediate values feeding the next layer.
- Data flows left to right in the **forward pass**; error information flows right to left in
  the **backward pass**.

**What do hidden units actually learn?** Nobody assigns them jobs. Training discovers that some
intermediate feature is useful and a unit drifts into computing it. That's the deal depth
offers: **learn useful intermediate representations, then solve an easy problem on top of them.**

Textbooks usually say the XOR hidden units learn clean logic gates like OR and AND. Here is
what the four units in *this* trained network actually compute (measured, not assumed —
exercise 7 shows you how to print this yourself):

```
  x1,x2      u1       u2       u3       u4
  0,0      0.532   -0.570    0.753   -0.342
  0,1     -0.697    0.762   -0.963    0.991
  1,0      0.994   -0.997   -0.977   -0.874
  1,1      0.903   -0.926   -1.000    0.939

  output layer weights:
          -2.243    2.737   -4.753   -4.449
```

Read it and the tidy story falls apart:

- **u1 and u2 are near-perfect mirror images** (u2 ≈ −u1 on every row). Two units learned the
  same feature with opposite sign — pure redundancy. Their output weights (−2.24 and +2.74)
  have opposite signs too, so they reinforce rather than cancel, acting as one feature with an
  effective weight near −5.
- **u3 is roughly NOR** — strongly positive only when both inputs are 0.
- **u4 roughly tracks x2** — positive whenever x2 = 1.

So it isn't OR-and-AND. It's *a* valid decomposition among many, found by gradient descent
from one random starting point, and a different seed produces a different one.

**This is the more useful lesson.** Learned representations are typically redundant, partially
duplicated, and only loosely interpretable. Reading meaning into individual hidden units is a
whole research area (*interpretability*) precisely because it's genuinely hard — the network is
under no obligation to organize itself the way a human would.

---

## 4. Learning = gradient descent

Now the core question: *how* do you adjust thousands of parameters in the right direction?

### The hill-descending picture

Imagine the loss as a landscape. Every parameter is a direction you can move in, and the
altitude is how wrong the network is. You want the lowest valley. You're in thick fog and can
only feel the slope under your feet.

The strategy: **feel which way is downhill, take a small step, repeat.**

That's gradient descent. The **gradient** is the slope — and calculus gives it to us exactly,
no guessing required:

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$

Read this as: *new weight = old weight − learning rate × slope of loss with respect to that weight.*

### Why the minus sign

The derivative `∂L/∂w` answers: "if I increase `w` slightly, does the loss go up or down?"

- Slope **positive** → increasing `w` increases loss → so **decrease** `w`.
- Slope **negative** → increasing `w` decreases loss → so **increase** `w`.

Both cases are handled by subtracting the gradient. The minus sign is what makes it *descent*.

### The learning rate η

How big a step to take.

- **Too small** → training crawls; thousands of wasted epochs.
- **Too large** → you overshoot the valley, bounce up the far side, and the loss explodes to
  `NaN`.
- **Just right** → steady decrease.

There's no formula for it. `0.1` to `0.5` is a reasonable starting range for a small network
like this one. If your loss ever goes to `NaN` or infinity, **lower the learning rate first** —
it's the cause the overwhelming majority of the time.

### One crucial caveat

Each parameter's derivative assumes **everything else stays fixed**. That's only true for an
infinitesimal step. Take a big step and all parameters move at once, and the terrain you
measured is no longer the terrain you're standing on. This is exactly why steps must be small
and why the process is iterative rather than one-shot.

---

## 5. Measuring wrongness: the loss function

The loss turns "how wrong was this prediction" into one number to minimize.

**Mean squared error (MSE)**, what this implementation uses:

$$L = \frac{1}{m}\sum_j (a_j - y_j)^2$$

Take the difference between prediction and target, square it, average over outputs. Squaring
does two useful things: it makes errors positive (so +0.3 and −0.3 are equally bad rather than
cancelling), and it punishes large errors disproportionately.

Its derivative — the seed of the entire backward pass:

$$\frac{\partial L}{\partial a_j} = \frac{2(a_j - y_j)}{m}$$

Sensible: if prediction exceeds target, the derivative is positive, meaning "lower this output."

MSE is the natural choice for regression (predicting numbers). For classification,
**cross-entropy** is better — see §25 item 3 and exercise 9.

---

## 6. The chain rule — the one piece of calculus you need

Backpropagation is the chain rule applied carefully. If you understand this section, the rest
is mechanical.

**The rule:** if `a` depends on `z`, and `z` depends on `w`, then

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**The intuition:** rates of change multiply along a chain. If a car is twice as fast as a
bike, and the bike is three times as fast as walking, the car is six times walking speed. The
same composition applies to "how much does nudging `w` move `L`."

In a neural network the chain is exactly:

```
w ──► z ──► a ──► (next layer) ──► ... ──► L
```

so the influence of any weight on the final loss is the product of every local rate of change
along the path from that weight to the loss.

The three factors in our case are all easy:

| Factor | What it is | Value |
|---|---|---|
| ∂L/∂a | How much loss changes with this unit's output | Comes from the loss, or from the layer above |
| ∂a/∂z | How much output changes with the weighted sum | `g'(z)` — the activation's derivative |
| ∂z/∂w | How much the weighted sum changes with this weight | `x` — just the input! |

That last one is worth pausing on. Since `z = w₁x₁ + w₂x₂ + b`, the derivative with respect
to `w₁` is simply `x₁`. **A weight's gradient is proportional to the input it multiplies.**
Which is intuitive: a weight attached to an input that was zero had no effect on this
prediction, so it gets no blame and no update.

---

## 7. Worked example: one neuron, by hand

Let's do a complete forward and backward pass with real numbers. *These values are computed
and verified — follow along with a calculator.*

**Setup.** One neuron, two inputs, sigmoid activation.

```
inputs   x = [1.0, 0.5]
weights  w = [0.3, -0.2]
bias     b = 0.1
target   y = 1.0
learning rate η = 0.5
```

> **A note on the `/m`.** §5 gave the loss derivative as `2(a - y)/m`, where `m` is the number
> of outputs. This network has one output, so `m = 1` and the division vanishes. That's why the
> examples below use plain `2(a - y)` — they aren't dropping a term, `m` just happens to be 1.
> The code in [`Network.cs`](src/NN/Network.cs) always divides properly.

### Forward pass

**Weighted sum:**

```
z = (0.3 × 1.0) + (-0.2 × 0.5) + 0.1
  = 0.3 - 0.1 + 0.1
  = 0.3
```

**Activation** (sigmoid: `1 / (1 + e⁻ᶻ)`):

```
a = 1 / (1 + e^-0.3) = 0.574443
```

**Loss:**

```
L = (a - y)² = (0.574443 - 1.0)² = (-0.425557)² = 0.181099
```

The network predicted 0.574 where 1.0 was wanted. Now we fix it.

### Backward pass

**Step 1 — how does loss change with the output?**

```
dL/da = 2(a - y) = 2 × (0.574443 - 1.0) = -0.851115
```

Negative, meaning: *increasing `a` would decrease the loss.* Correct — we want `a` to grow
toward 1.0.

**Step 2 — push back through the activation.** Sigmoid's derivative is `a(1-a)`:

```
g'(z) = a(1 - a) = 0.574443 × 0.425557 = 0.244458

delta = dL/da × g'(z) = -0.851115 × 0.244458 = -0.208062
```

`delta` (δ) is the single most important intermediate value in backprop. It means:
**"how much does the loss change per unit change in this neuron's `z`."** Once you have δ for
a neuron, every gradient it's involved in is one multiplication away.

**Step 3 — the parameter gradients.** Multiply δ by each input:

```
dL/dw₀ = delta × x₀ = -0.208062 × 1.0 = -0.208062
dL/dw₁ = delta × x₁ = -0.208062 × 0.5 = -0.104031
dL/db  = delta × 1   = -0.208062
```

Notice `w₁`'s gradient is **half** of `w₀`'s — because its input was 0.5 instead of 1.0. It
had half the influence, so it gets half the correction. The bias gradient is just δ, since a
bias is a weight on a constant input of 1.

### The update

Subtract `η × gradient` from each parameter:

```
w₀ = 0.3  - 0.5 × (-0.208062) =  0.404031
w₁ = -0.2 - 0.5 × (-0.104031) = -0.147984
b  = 0.1  - 0.5 × (-0.208062) =  0.204031
```

### Did it work?

Run the forward pass again with the new parameters:

```
z = (0.404031 × 1.0) + (-0.147984 × 0.5) + 0.204031 = 0.534070
a = sigmoid(0.534070) = 0.630432
L = (0.630432 - 1.0)² = 0.136581
```

**Loss dropped from 0.181099 to 0.136581**, and the prediction moved from 0.574 to 0.630 —
closer to the target of 1.0.

That is a full training step. A network is this repeated across every neuron and every
example, thousands of times.

---

## 8. Worked example: the chain across two layers

The above handled one layer. The essential new idea in a *deep* network is propagating the
gradient **backwards into the previous layer**. Here's the smallest possible case, also verified.

**Setup.** One input → one tanh hidden unit → one sigmoid output unit.

```
x = 0.8      w₁ = 0.5, b₁ = 0.0   (hidden, tanh)
             w₂ = 1.2, b₂ = -0.3  (output, sigmoid)
target y = 0.0
```

### Forward

```
z₁ = 0.5 × 0.8 + 0.0 = 0.400000
h  = tanh(0.4)       = 0.379949      ← hidden unit's output

z₂ = 1.2 × 0.379949 - 0.3 = 0.155939
a  = sigmoid(0.155939)    = 0.538906  ← final prediction

L  = (0.538906 - 0.0)² = 0.290420
```

### Backward — output layer first

```
dL/da  = 2(a - y) = 2 × 0.538906 = 1.077812
sig'   = a(1-a)   = 0.538906 × 0.461094 = 0.248486
delta₂ = 1.077812 × 0.248486 = 0.267821

dL/dw₂ = delta₂ × h = 0.267821 × 0.379949 = 0.101758
dL/db₂ = delta₂                            = 0.267821
```

### Backward — now cross into the hidden layer

**This is the step that makes it "backpropagation."** The hidden unit influenced the loss only
*through* the output unit, so we route the gradient back through the connecting weight `w₂`:

```
dL/dh = delta₂ × w₂ = 0.267821 × 1.2 = 0.321386
```

Read that as: *"the hidden unit's output affects the loss by δ₂ scaled by the strength of the
connection carrying it forward."* The same weight `w₂` that carried the signal **forward** now
carries the blame **backward**. That symmetry is the heart of the algorithm.

Then it's the identical recipe as before, one layer down:

```
tanh'  = 1 - h² = 1 - 0.144361 = 0.855639
delta₁ = dL/dh × tanh' = 0.321386 × 0.855639 = 0.274990

dL/dw₁ = delta₁ × x = 0.274990 × 0.8 = 0.219992
dL/db₁ = delta₁                       = 0.274990
```

### The pattern

Every layer, without exception, does:

1. Receive `dL/da` from above (or from the loss, if it's the last layer).
2. `δ = dL/da × g'(z)` — push through the activation.
3. `dL/dW += δ × input`, `dL/db += δ` — record the parameter gradients.
4. `dL/dinput = δ × W` — hand backwards to the layer below, which returns to step 1.

**With 2 layers or 200, that loop is the entire algorithm.** [`Dense.Backward`](src/NN/Dense.cs#L132)
is a literal transcription of those four lines.

### Why it's efficient

Naively you might compute each weight's gradient separately by re-running the network — with
a million weights that's a million forward passes. Backprop computes **all** gradients in a
single backward sweep by reusing the shared δ values. It costs roughly the same as one forward
pass. That efficiency is why neural networks are trainable at all, and it's why the 1986
popularization of backprop restarted the entire field.

---

## 9. The perceptron, XOR, and why this history matters

The **perceptron** (1958) is the ancestor of all this: one unit, step activation, and its own
update rule.

$$w \mathrel{+}= \eta\,(y - \hat{y})\,x$$

Since the step function outputs only 0 or 1, the error `(y - ŷ)` is always −1, 0, or +1:
"too high," "correct," or "too low." [`Perceptron.Train`](src/NN/Perceptron.cs#L29) skips correct predictions
entirely and stops early once a full epoch passes with no updates.

**The convergence theorem:** if the data is *linearly separable* — separable by a single
straight line — this is guaranteed to converge in finite steps. If it isn't, it oscillates forever.

```
      AND (separable)              XOR (not separable)

  1 │  ○           ●           1 │  ●           ○
    │      ╲                     │
    │        ╲                   │        no single straight line
  0 │  ○      ╲  ○             0 │  ○           ●    separates ● from ○
    └───────────────           └───────────────
      0           1               0           1
```

Minsky and Papert's 1969 book *Perceptrons* analyzed this limitation rigorously, and it's
widely credited with cooling enthusiasm for neural networks through the 1970s.

*Treat that story with some caution* — it's repeated everywhere in simplified form. Minsky and
Papert did discuss multilayer networks rather than ignoring them, and historians generally
attribute the broader "AI winter" funding collapse more to the 1973 Lighthill report and
unmet expectations across AI as a whole. The mathematical result is solid; the tidy
cause-and-effect narrative around it is contested.

**The fix is the two things this codebase adds:** a hidden layer (so the network can build its
own features) and a smooth activation (so backprop can compute gradients through it).

You can watch the whole story run in [`Program.cs`](src/NN.Demo/Program.cs): the perceptron nails AND in
4 epochs, and the network with one hidden layer solves XOR that the perceptron provably cannot.

---
---

# Part II — The C# Implementation

---

## 10. Reading order

If you're reading the code for the first time, go in this order:

1. **[`Activations.cs`](src/NN/Activations.cs)** — activations. Small, self-contained, and each one matches the table in §13.
2. **[`Dense.cs`](src/NN/Dense.cs) → `Forward`** — §2's math, literally.
3. **[`Dense.cs`](src/NN/Dense.cs) → `Backward`** — §8's four-step pattern, literally.
4. **[`Network.cs`](src/NN/Network.cs)** — the training loop that drives it all.
5. **[`SimdOps.cs`](src/NN/SimdOps.cs)** — pure optimization. **Skip on the first pass**; it changes
   nothing conceptually.

---

## 11. C# features used here that you may not have met

These are the constructs that make the code look unfamiliar. None are essential to the math.

### `Span<float>` — a window into memory

```csharp
public void Forward(ReadOnlySpan<float> aIn, Span<float> aOut)
```

A `Span<float>` is a **view** of a slice of an array — a pointer plus a length. It copies
nothing. `weights.AsSpan(10, 4)` refers to elements 10–13 of the original array; writing
through the span writes the original.

Why it's everywhere here: training calls `Forward` millions of times. Returning a new `float[]`
each call would allocate millions of arrays and bury you in garbage collection. Spans let one
pre-allocated buffer be reused forever. `ReadOnlySpan` additionally documents "I only read this."

The catch: a span can't be stored in a class field or used inside `async` methods (it's a
`ref struct`, stack-only). That's why the layers keep `float[]` fields and hand out spans.

### Generic structs as policy — `Dense<TActivation>`

```csharp
public sealed class Dense<TActivation> : ILayer where TActivation : IActivation
```

The activation is a **type parameter**, not a field. `Dense<Tanh>` and `Dense<Sigmoid>` are
different types, and the JIT compiles separate machine code for each, inlining the activation
directly into the loop.

The obvious alternative — a `Func<float, float>` field — costs an indirect call for every unit
of every layer of every example, and the compiler can't inline through it. This pattern gets
the flexibility for free.

The enabling feature is **static abstract interface members** (C# 11):

```csharp
public interface IActivation
{
    static abstract float Apply(float z);
}
```

An interface method with no object attached. It lets `TActivation.Apply(z)` be resolved and
inlined at compile/JIT time.

### `readonly struct` activations

`Sigmoid`, `Tanh`, and the rest are empty structs. They hold no data and are never
instantiated — they exist purely as *names* to feed the generic system. Zero runtime cost.

### The JIT

C# compiles to IL (bytecode), and the JIT compiler turns IL into machine code at runtime. It
inlines small methods, picks SIMD instructions for the actual CPU, and — importantly here —
generates a **separate code copy per value-type generic instantiation**. That's what makes
`Dense<Tanh>` fast, and also why §14 keeps the SIMD helpers *out* of the generic class.

---

## 12. Data layout — the most consequential decision

NumPy stores `W` as `(n, j)`: **n features × j units**. Unit `j`'s weights are the *column*
`W[:, j]`, whose elements sit `n` floats apart in memory.

This C# version stores the **transpose**, flattened into one array:

```
Weights = [ unit0: w00 w01 w02 | unit1: w10 w11 w12 | unit2: ... ]
            └── Inputs floats ──┘
```

Unit `j`'s weights are `Weights[j * Inputs .. (j+1) * Inputs]` — **contiguous**.
See [`Dense.UnitWeights`](src/NN/Dense.cs#L62).

### Why this matters so much

A CPU never loads one float. It loads a **64-byte cache line** (16 floats) at a time, because
memory is vastly slower than arithmetic — a main-memory fetch costs hundreds of cycles while a
multiply costs one.

- **Strided (NumPy column) access:** load 16 floats, use 1, discard 15. You waste ~94% of your
  memory bandwidth, and SIMD can't be used without an expensive gather instruction.
- **Contiguous access:** load 16 floats, use all 16, and a single SIMD instruction processes
  4–16 of them at once (§15).

> **Rule of thumb:** arrange data so the innermost loop walks consecutive addresses. This one
> habit is worth more than every other optimization in this file combined.

**One flat array, not jagged.** `float[]` beats `float[][]` here: one allocation instead of
`j`, one bounds check instead of two, no pointer chase per unit, and the whole matrix sits in
one contiguous block the prefetcher can stream.

---

## 13. Activations in code

| Activation | g(z) | g'(z) in terms of a = g(z) | Range | Use for |
|---|---|---|---|---|
| **Sigmoid** | 1 / (1 + e⁻ᶻ) | `a(1 - a)` | (0, 1) | Output layer, binary probability |
| **Tanh** | tanh(z) | `1 - a²` | (−1, 1) | Hidden layers |
| **ReLU** | max(0, z) | `a > 0 ? 1 : 0` | [0, ∞) | Hidden layers, deep nets |
| **Identity** | z | `1` | ℝ | Regression output |
| **Step** | z ≥ 0 ? 1 : 0 | *throws* | {0, 1} | Perceptron only |

### Why derivatives are written in terms of `a`, not `z`

Every activation here has a derivative recoverable from **its own output**. Sigmoid's
derivative is `a(1-a)`; tanh's is `1-a²`. The forward pass already computed `a`, so the
backward pass reuses it instead of recomputing `exp()` or `tanh()`.

This is a real saving, not a micro-optimization: transcendental functions cost 20–100× a
multiply. It's also why `Forward` caches its outputs — see §14.

### Why `Step` throws instead of returning 0

Its derivative is 0 everywhere it's defined (the function is flat), and undefined at the jump.
Returning 0 would multiply every gradient in the network by zero, and training would appear to
run while learning **nothing** — a silent, maddening bug.

Throwing makes the lesson explicit: **the step function's flatness is precisely why the
perceptron needs its own update rule, and why backprop requires smooth activations.** Gradient
descent needs a slope to follow; a staircase has none.

### Choosing one

- **Hidden layers → tanh or ReLU.** Tanh is zero-centered, meaning outputs span (−1, 1).
  Sigmoid's outputs are all positive, which makes every weight gradient into a unit share a
  sign, so optimization zigzags instead of heading straight downhill.
- **Output layer → match the task.** Sigmoid for a probability in (0,1); identity for
  unbounded regression.
- **Sigmoid and tanh saturate:** for large |z| the curve flattens, `g'(z) ≈ 0`, and gradients
  nearly vanish. Stack many such layers and the gradient reaching the early layers is
  effectively zero — the **vanishing gradient problem**. ReLU's derivative is exactly 1 for
  positive inputs, which is why deep networks moved to it.

---

## 14. The forward pass in code

[`Dense.Forward`](src/NN/Dense.cs#L89):

```csharp
for (int j = 0; j < Units; j++)
{
    float z = SimdOps.Dot(w.Slice(j * Inputs, Inputs), aIn) + Bias[j];
    aOut[j] = TActivation.Apply(z);
}

aIn.CopyTo(_lastInput);     // cached for backprop
aOut.CopyTo(_lastOutput);
```

Line for line, this is §2: slice the unit's weights, dot with the input, add bias, activate.
Compare it to the original NumPy and the correspondence is exact — only the layout changed.

**Why the caching at the bottom.** Backprop needs both the inputs that produced these
activations (for `dL/dW = δ × input`) and the activations themselves (for `g'` from `a`).

This is the memory cost of training, and it's worth understanding: **you cannot free forward
activations until the backward pass has consumed them.** It's the main reason training a large
model needs far more memory than running one, and why "batch size too large" means
out-of-memory in practice.

---

## 15. SIMD — doing several multiplications at once

**Skip this section on a first read.** It's pure speed; the math is unchanged.

SIMD = *Single Instruction, Multiple Data*. Modern CPUs have wide registers holding several
floats at once, and one instruction operates on all of them simultaneously.

**How many depends on your CPU:**

| Hardware | `Vector<float>.Count` |
|---|---|
| Apple Silicon / ARM (NEON) | **4** |
| x86 with AVX2 | 8 |
| x86 with AVX-512 | 16 |

`Vector<float>` from `System.Numerics` exposes this portably — you write the code once and it
adapts. Check your own machine with:

```csharp
Console.WriteLine(System.Numerics.Vector<float>.Count);
```

This is exactly why the code never hardcodes a width: every loop in [`SimdOps.cs`](src/NN/SimdOps.cs)
reads `Vector<float>.Count` at runtime, so the same source runs at full width on both ARM and
x86.

> **Common misconception, and the one that started this project's rewrite:** `Vector<float>` is
> **not** a variable-length mathematical vector. It's a fixed-width hardware register. It cannot
> store a layer's weights — it's a pipe you stream data through, `Count` floats at a time.

### The dot product ([`SimdOps.Dot`](src/NN/SimdOps.cs#L15))

```csharp
var acc0 = Vector<float>.Zero;
var acc1 = Vector<float>.Zero;

for (; i <= n - 2 * width; i += 2 * width)
{
    acc0 += new Vector<float>(a.Slice(i, width))         * new Vector<float>(b.Slice(i, width));
    acc1 += new Vector<float>(a.Slice(i + width, width)) * new Vector<float>(b.Slice(i + width, width));
}
// ... then a single-width loop, a horizontal sum, and a scalar tail
```

**Why two accumulators.** A multiply-add instruction has ~4 cycles of *latency* but can be
*issued* every cycle. With a single accumulator, each iteration must wait for the previous
one's result — you get one result per 4 cycles, a quarter of peak. Two independent chains let
the CPU keep four operations in flight. This is **instruction-level parallelism**, and it costs
one extra register.

**The scalar tail** handles a length that isn't a multiple of the SIMD width. Every
hand-written SIMD loop needs one; forgetting it silently drops the last few elements.

### `AddScaled` — `dest += src × scale`

[`SimdOps.AddScaled`](src/NN/SimdOps.cs#L47) is the workhorse of the *backward* pass. Notice from §8
that steps 3 and 4 are both "add a scaled vector into an accumulator" — the same primitive
serves weight-gradient accumulation, input-gradient propagation, and the descent step itself.

### Why these live outside `Dense<T>`

The JIT emits a separate code copy per value-type generic instantiation (§11). Left inside
`Dense<TActivation>`, `Dot` would be duplicated for `Dense<Tanh>`, `Dense<ReLU>`,
`Dense<Sigmoid>`… bloating the instruction cache for no benefit. A non-generic helper class
gets exactly one copy.

---

## 16. The backward pass in code

[`Dense.Backward`](src/NN/Dense.cs#L132) — §8's four-step pattern, transcribed:

```csharp
for (int j = 0; j < Units; j++)
{
    float delta = gradOut[j] * TActivation.DerivativeFromOutput(a[j]);   // step 2
    if (delta == 0f) continue;                                          // dead ReLU shortcut

    int offset = j * Inputs;
    SimdOps.AddScaled(_weightGrads.AsSpan(offset, Inputs), x, delta);    // step 3: dL/dW += δ·x
    _biasGrads[j] += delta;                                             // step 3: dL/db += δ

    if (propagate)
        SimdOps.AddScaled(gradIn, Weights.AsSpan(offset, Inputs), delta); // step 4: dL/dx += δ·W
}
```

Three things worth noticing:

**Steps 3 and 4 are the same operation with weights and inputs swapped.** `dL/dW += δ·x` and
`dL/dx += δ·W` are mirror images — that symmetry is why one `AddScaled` primitive covers both.
And thanks to the layout from §12, *both* walk memory contiguously. The decision that sped up
the forward pass pays off twice more here.

**`gradIn` is empty for the first layer.** Nothing consumes the gradient with respect to the
raw input data, so computing it would be wasted work. (In some techniques — adversarial
examples, style transfer — that input gradient is exactly what you want. Not here.)

**`if (delta == 0f) continue;`** skips units contributing nothing. With ReLU this is common:
any unit whose output was clamped to 0 has a zero derivative, so it accumulates nothing. Which
also flags a real failure mode — see §23.

### Accumulate now, apply later

`Backward` only **adds into** `_weightGrads`. It never touches `Weights`.
[`ApplyGradients`](src/NN/Dense.cs#L165) performs the actual descent step and clears the accumulators:

$$W \mathrel{-}= \eta \cdot \frac{1}{\text{batchSize}}\frac{\partial L}{\partial W}$$

Why separate them? Because it lets you sum gradients over several examples before updating —
mini-batching, §20. Dividing by batch size averages rather than sums, so your learning rate
keeps working when you change batch size.

This split is not an idiosyncrasy of this code. PyTorch divides at exactly the same seam:
`loss.backward()` accumulates, `optimizer.step()` applies. If you move to a real framework,
this will look familiar.

---

## 17. Weight initialization — genuinely not optional

[`Dense.Initialize`](src/NN/Dense.cs#L75) uses **Xavier/Glorot uniform**:

$$W \sim \mathcal{U}\left(-\sqrt{\tfrac{6}{n_{in} + n_{out}}},\; +\sqrt{\tfrac{6}{n_{in} + n_{out}}}\right)$$

### Why identical weights fail — the symmetry problem

Start every weight in a layer at the same value and every unit computes the identical output.
Identical outputs earn identical gradients. Identical gradients produce identical updates.
**The units stay identical forever.**

A 100-unit layer initialized this way has the expressive power of *one* unit, permanently — no
amount of training escapes it, because nothing breaks the tie. Random initialization **breaks
the symmetry**: each unit starts different, receives a different gradient, and specializes.

### Why all-*zero* weights fail even harder

Zero is a special case, and it's worth separating because it's the one you can run (exercise 2).
It doesn't merely collapse the layer to one effective unit — it stops learning **entirely**:

- Every hidden unit outputs `g(0)`, and the gradient handed back to the hidden layer is
  `δ·W = δ·0 = 0`.
- The hidden weight gradient is `δ·x` with `δ = 0`, so hidden weights never move.
- With `tanh`, the hidden outputs are `tanh(0) = 0`, so the output layer's weight gradients
  (`δ·h`) are zero too.

Nothing but the output bias can move, and on XOR even that stays put — the four examples'
gradients cancel exactly. Run exercise 2 and you'll see loss frozen at **exactly 0.250000**
with every prediction 0.5000: the network permanently guessing the mean, having learned
literally nothing.

**Biases can safely start at zero** — the random weights already broke the symmetry.

### Why that particular scale

Too large and activations saturate at ±1 where gradients vanish; too small and the signal
decays toward zero as it passes through layers. Xavier's `6/(fan_in + fan_out)` keeps
activation variance roughly constant across layers, so signals neither explode nor vanish with
depth. For ReLU, **He initialization** (variance `2/fan_in`) is the better-matched choice, since
ReLU discards half its input range.

---

## 18. The perceptron in code

[`Perceptron`](src/NN/Perceptron.cs) is the odd one out: it's the only thing here that doesn't
use backpropagation. It exists to make §9's history concrete and runnable.

It's a `Dense<Step>` of exactly one unit, wrapped in its own training rule:

```csharp
public float Predict(ReadOnlySpan<float> x)
{
    _layer.Forward(x, _out);
    return _out[0];
}
```

The training loop ([`Perceptron.Train`](src/NN/Perceptron.cs#L29)) is the 1958 rule verbatim:

```csharp
float error = y[i] - Predict(xi);
if (error == 0f) continue;          // correct prediction: no update at all

float delta = learningRate * error;
for (int k = 0; k < Inputs; k++)
    w[k] += delta * xi[k];
Bias += delta;
```

Three things to notice, all of which contrast instructively with backprop:

**No derivative appears anywhere.** Compare with `Dense.Backward`, where every gradient is
multiplied by `g'(z)`. The perceptron rule doesn't need one — which is exactly as well, because
`Step.DerivativeFromOutput` throws (§13). This is the same fact from two directions: a step
function has no usable slope, so gradient descent can't work on it, and the perceptron sidesteps
that by not being gradient descent.

**`error` is only ever −1, 0, or +1**, since both prediction and target are 0 or 1. It's a
direction — "too high", "correct", "too low" — not a magnitude. Backprop's δ carries magnitude
as well, which is what lets it say *how much* to correct.

**Correct predictions cause no update**, and an epoch with no updates ends training early. That
early exit is what makes "converged in 4 epochs" a meaningful statement rather than just the
epoch limit being hit — and it's why the XOR case runs the full budget instead.

The tests pin both behaviours: `Perceptron_converges_on_linearly_separable_data` and
`Perceptron_cannot_learn_xor`. The second asserts *failure*, which is unusual and deliberate —
it locks in a property of the algorithm that the rest of the library exists to overcome.

---

## 19. The network, and the Sequential builder

[`Network`](src/NN/Network.cs) holds `ILayer[]`. This is a **sequential** model in exactly the Keras
sense: a strictly linear chain, each layer's output feeding the next, no branching.

### Building one

[`Sequential`](src/NN/Sequential.cs) is a fluent builder that mirrors the Keras API you'll meet in
courses and tutorials:

```csharp
var net = new Sequential(inputs: 2)
    .Dense<Tanh>(4)
    .Dense<Sigmoid>(1)
    .Build(seed: 42);
```

```python
# the Keras equivalent
model = Sequential([
    Dense(4, activation='tanh'),
    Dense(1, activation='sigmoid'),
])
```

The builder's real job is **input-size inference**. You declare the input width once; each
layer takes its input count from the previous layer's unit count. The direct constructor still
works and does the same thing, but makes you state — and keep consistent — both ends of every
layer:

```csharp
var net = new Network(seed: 42,
    new Dense<Tanh>(inputs: 2, units: 4),
    new Dense<Sigmoid>(inputs: 4, units: 1));   // the 4 must match by hand
```

Two shape errors are impossible to express through the builder and merely caught by the
constructor: `.Add(layer)` rejects a layer whose `Inputs` doesn't match the current width, and
`Build()` rejects an empty stack.

### `Summary()`

Like Keras' `model.summary()`:

```
Layer                     Output    Params
──────────────────────────────────────────
Dense<Tanh>                    4        12
Dense<Sigmoid>                 1         5
──────────────────────────────────────────
Input width: 2
Trainable parameters: 17
```

Check the arithmetic yourself — it's the formula from §3. The tanh layer has 2 inputs × 4 units
= 8 weights, plus 4 biases = 12. The output layer has 4 × 1 = 4 weights plus 1 bias = 5.

### Saving and loading a trained model

Training produces one thing of value: the weights. [`ModelIO`](src/NN/ModelIO.cs) writes them, together
with enough architecture to rebuild the network:

```csharp
ModelIO.Save(net, "xor.nnm");            // after training
var loaded = ModelIO.Load("xor.nnm");    // later, or in another program
float p = loaded.Predict(input)[0];      // ready to use immediately — no training needed
```

The XOR model is **127 bytes**, and the reload is bit-for-bit identical: same inputs, same
outputs, exactly.

**The file format**, little-endian:

```
  magic     8 bytes   "NNMODEL\0"
  version   int32     1
  layers    int32     how many
  per layer:
    descriptor  string   "Dense<Tanh>"
    inputs      int32
    units       int32
    weights     float32 × inputs × units
    biases      float32 × units
```

Four decisions in there are worth understanding, because they're the ones people get wrong:

**Save the architecture, not just the weights.** A bare float dump has no idea what shape it
was, so loading it into mismatched code silently produces garbage. Storing layer types and
shapes means a mismatch is caught immediately.

**Magic bytes and a version number.** Without them, any wrong file gets interpreted as floats and
"works" until the predictions turn out to be nonsense. With them, you get *"Not a model file"* or
*"Model format version 99 is not supported"*. Truncated files are caught too, naming the layer
that ran out of data. **Fail loudly on bad input** is the general lesson; model files are just a
place where it's easy to skip.

**A descriptor→constructor table** ([`ModelIO.Factories`](src/NN/ModelIO.cs)) turns the string
`"Dense<Tanh>"` back into a real type. C# can't construct a generic type from a string without
either this table or reflection, and an explicit table is both faster and safer — reflection would
let a malicious file name *any* type in your process. `ModelIO.Register` extends it for custom layers.

**Loading must not initialize.** The public `Network` constructor randomizes weights, so
constructing a network from loaded layers would destroy the parameters you just read. That's why
`ModelIO` calls `Network.FromTrainedLayers`, which skips initialization. It's a real bug I hit
writing this, and it's nasty precisely because it doesn't crash — you'd just get an untrained
network that loads "successfully."

> **What is *not* saved:** gradient accumulators and the forward-pass activation cache. Those are
> training scratch space, rebuilt empty on load. If you later add momentum or Adam, their state
> would also need saving to resume training mid-run — see §25 item 7.

### Where sequential stops being enough

An array walked front-to-back can only express a chain. Skip connections (ResNet), multiple
inputs or outputs, and concatenation all need a **graph** of layers with a topological sort
instead of these two loops — that's what Keras' functional API is for. Everything in an
introductory course is sequential, which is why Keras makes it the default.

### The layer stack itself

**Why a non-generic interface exists alongside the generics.** `Dense<Tanh>` and
`Dense<Sigmoid>` are *unrelated types* — you cannot put them in the same array. Generics give
speed; the [`ILayer`](src/NN/ILayer.cs) interface gives heterogeneity. You need both, so the code has
both.

**Forward:** `x → layer0 → layer1 → … → output`

**Backward:** walk the array in reverse, each layer's `gradIn` becoming the next one's `gradOut`:

```csharp
for (int i = last; i >= 0; i--)
    _layers[i].Backward(_grads[i], i > 0 ? _grads[i - 1] : Span<float>.Empty);
```

That single line is §8's "hand the gradient backwards" step, generalized to any depth.

The constructor validates that adjacent layer shapes match — layer `i`'s `Inputs` must equal
layer `i-1`'s `Units` — and fails immediately with a clear message rather than producing
nonsense later. It also pre-allocates every buffer, so training allocates nothing per example.

---

## 20. The training loop

Per **epoch** (one full pass over the data):

1. **Shuffle** the example order.
2. For each **mini-batch**: accumulate gradients over its examples, then apply once.

### Why shuffle

With a fixed order the network can learn the *order* rather than the data, and consecutive
correlated samples (all of class A, then all of class B) yield biased gradient estimates that
make training lurch. Shuffling makes each batch a fairer sample of the whole dataset.

### Batch size — the three regimes

| Batch size | Name | Behavior |
|---|---|---|
| 1 | Stochastic (SGD) | Fast, noisy updates. Noise can help escape shallow local minima. |
| 8–256 | Mini-batch | The standard compromise, and far more hardware-efficient. |
| All | Full batch | Smoothest gradient, slowest progress, most memory. |

The default here is full batch, since XOR has four examples.

Averaging over a batch reduces gradient noise — individual examples disagree about the best
direction, and averaging finds their consensus.

### Epochs

One epoch = one pass over all examples. Networks need many because each update is a small step
(§4).

XOR takes ~4000 epochs here, which sounds enormous. But the demo runs **full batch**, so one
epoch = one parameter update: 4,000 actual gradient-descent steps, each informed by all four
examples (16,000 example evaluations). Four thousand small steps to tune 17 parameters is
unremarkable — and it's why the distinction between an *epoch* and an *update* matters. At
batch size 1 the same 4000 epochs would mean 16,000 updates.

---

## 21. Gradient checking — how you *know* the code is right

This is the section most tutorials omit, and it's the most practically valuable one.

**The problem:** a subtly wrong backward pass usually still trains *somewhat*. Loss goes down,
nothing crashes, and results are merely mediocre — indistinguishable from "needs tuning." These
bugs can cost days.

**The defense:** compare against a numerical estimate that makes no use of your backprop code
([`GradientCheck.cs`](src/NN/GradientCheck.cs)). Nudge one weight up and down, measure how the loss
actually moves, and compare:

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$$

This is the definition of a derivative with a finite ε instead of a limit. It's far too slow
for training — two full forward passes *per parameter* — but perfect for verification. The
**central** difference (both directions) has O(ε²) error versus O(ε) for the one-sided version,
which easily repays the second evaluation.

### The measured result, and why it proves correctness

| ε | max relative error | dominated by |
|---|---|---|
| 1e-1 | 9.1e-3 | truncation — ε too coarse to be a good derivative |
| **1e-2** | **2.4e-4** | balanced ← best |
| 1e-3 | 1.7e-3 | roundoff creeping in |
| 1e-4 | 1.5e-2 | float32 roundoff — `L(w+ε)` and `L(w−ε)` nearly identical |

**The U-shape *is* the proof.** Two error sources fight each other: large ε is a poor
approximation of a derivative, while small ε subtracts two nearly-equal floats and loses
precision catastrophically. A correct gradient shows this tradeoff with a sweet spot in the
middle. **A wrong gradient shows O(1) error at every ε** — no sweet spot, because it isn't
approximating anything.

`float` bottoms out around 1e-4; production checks use `double` and expect ~1e-10.

**Rule: whenever you add a layer type or activation, gradient-check it before trusting it.**

### Using it

```csharp
float error = GradientCheck.MaxRelativeError(net, oneInput, itsTarget);
```

It works on any network because layers expose their parameters through a flat, type-agnostic
index (`GetParameter` / `SetParameter` / `GetParameterGradient` on [`ILayer`](src/NN/ILayer.cs))
rather than the check knowing about `Dense` specifically.

### In the test suite

The checks run as real tests (`dotnet test`), including two that are easy to overlook:

- **A deliberately broken derivative** — a `Tanh` variant using `1 + a²` instead of `1 - a²` —
  must produce O(1) error. Without this, the passing tests might be passing *vacuously*: a check
  that cannot fail proves nothing.
- **Depth raises the error floor.** Three layers measure ~4.6e-3 against ~2.4e-4 for two, because
  each extra layer compounds float32 roundoff in the loss evaluations the finite difference
  depends on. The threshold is loosened for the deeper test — but it still separates a correct
  gradient from a broken one by more than an order of magnitude.

---
---

# Part III — Practice

---

## 22. Results

```
Perceptron on AND: converged in 4 epochs      ← linearly separable

Network on XOR (2 -> 4 tanh -> 1 sigmoid):
  epoch  1000  loss 0.002304
  epoch  4000  loss 0.000350
  0 XOR 0 -> 0.0100    1 XOR 0 -> 0.9779
  0 XOR 1 -> 0.9826    1 XOR 1 -> 0.0226      ← the hidden layer earning its keep

Gradient check: max relative error = 3.521E-004
```

The XOR outputs aren't exactly 0 and 1 because sigmoid only *approaches* its limits — reaching
1.0 exactly would need infinite weights. 0.98 means "confidently 1."

---

## 23. Debugging playbook

The failure modes you'll actually hit, and what they mean:

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss → `NaN` or ∞ | Learning rate too high; weights exploding | Divide learning rate by 10 |
| Loss flat at a middling value | Zero/constant init — symmetry unbroken (§17) | Random initialization |
| Loss decreases then plateaus high | Not enough capacity, or saturated units | More hidden units; try tanh over sigmoid |
| Loss barely moves | Learning rate too low, or vanishing gradients | Raise learning rate; use ReLU/tanh |
| Trains but predicts poorly | Gradient bug, or genuinely too few epochs | **Gradient check first** (§21) |
| ReLU net stops improving | Dead units — output 0 for all inputs, so gradient is permanently 0 | Lower learning rate; try leaky ReLU |
| Perfect on training data, bad on new data | Overfitting — memorization, not learning | More data, fewer parameters, regularization |

**Always gradient-check before tuning hyperparameters.** Tuning a buggy gradient is an
unbounded waste of time.

---

## 24. Exercises

Worked roughly in order of value. The "break it" ones teach the most.

1. **Watch XOR fail.** Point `Perceptron` at the XOR data instead of AND. It won't converge —
   that's the Minsky–Papert wall from §9, and feeling it beats reading about it.
2. **Break initialization.** Add `Array.Clear(Weights); return;` at the top of
   `Dense.Initialize` so weights stay zero. XOR freezes at loss **exactly 0.250000**, predicting
   0.5000 for every input, forever. That's §17's zero-weight deadlock — the most convincing
   demonstration here of why initialization matters. Then try initializing every weight to the
   same *nonzero* value (say 0.5) and watch the different, milder failure: the layer learns, but
   as though it had a single unit.
3. **Break the gradient.** Change `Tanh.DerivativeFromOutput` to `1 + a * a`. Watch the gradient
   check jump to O(1) while training *still partly works*. This is why §21 exists.
4. **Sweep the learning rate.** Try 0.01, 0.1, 0.5, 2.0, 10.0 on XOR. You'll see slow crawling,
   healthy descent, and divergence to `NaN` — the whole spectrum from §4.
5. **Shrink the hidden layer** to 1 unit. XOR becomes unsolvable again; 2 is the theoretical
   minimum. Find where it starts working reliably.
6. **Try ReLU** in the hidden layer. It may need a lower learning rate. Print each hidden unit's
   output across all four inputs to spot dead units.
7. **Reproduce and then perturb the hidden-feature table in §3.** Cast `net.Layers[0]` to
   `Dense<Tanh>`, call `Forward` on each of the four inputs, and print the results — you should
   get exactly the numbers in §3. Now change the network's seed (`new Network(seed: 7, …)`) and
   print again. You'll get a completely different, equally valid decomposition. This is the most
   illuminating exercise here: it shows there's no single "correct" set of learned features.
8. **Add momentum:** keep a velocity buffer per layer, `v = βv + grad` (β ≈ 0.9), and step along
   `v`. Compare convergence speed.
9. **Add cross-entropy loss.** For classification it beats MSE: with a sigmoid output the
   `σ'(z)` factor cancels against the loss derivative, removing the slowdown that occurs when
   the network is confidently wrong (where `σ'` ≈ 0 nearly kills the gradient).
10. **Implement `ForwardBatch` as a real tiled GEMM** (§25, item 1) and benchmark it.

---

## 25. What this implementation does *not* do

Honest limits, ordered by how much they cost:

1. **No GEMM batching.** `ForwardBatch` just loops the single-example path, re-streaming the
   whole weight matrix per example. Single-example inference is **memory-bandwidth bound** —
   about one multiply-add per float loaded. Batching into a *tiled matrix-matrix multiply*
   reuses each loaded weight block across many examples, raising arithmetic intensity by
   roughly the batch size. Tuned BLAS libraries commonly report order-of-magnitude gains from
   this, though **nothing here has been benchmarked** — exercise 10 is where you'd measure it
   rather than take my word. Expect it to dominate everything else on this list, including all
   the SIMD work.
2. **Plain SGD.** No momentum, Adam, learning-rate schedule, or weight decay. Adam typically
   converges several times faster.
3. **MSE only.** Cross-entropy is the right loss for classification (exercise 9).
4. **Single-threaded.** No `Parallel.For` over units or batch rows.
5. **No explicit FMA.** `acc += a * b` may or may not fuse into one instruction;
   `Vector256.FusedMultiplyAdd` guarantees it, at the cost of writing a separate ARM path.
6. **No regularization, validation split, or early stopping** — so nothing detects overfitting.
7. **Serialization saves parameters only** — not optimizer state (there is none yet) or training
   history. Fine for inference; you cannot resume training mid-run from a file.

For production, the fastest C# is C# that calls something else: `TensorPrimitives`, ONNX
Runtime, or a GPU. Nobody beats a tuned BLAS with hand-written loops. This code exists for
understanding, and at this scale it will never be your bottleneck.

---

## 26. Where to go next

Roughly in order:

1. **Cross-entropy loss + softmax output** — the standard classification setup.
2. **A real dataset.** MNIST (28×28 handwritten digits) is the classic first one: a
   784 → 128 → 10 network trains in minutes and gets ~97%.
3. **Momentum, then Adam.**
4. **Regularization** — dropout, weight decay — once you can overfit something.
5. **Convolutional layers**, if images interest you.
6. **A real framework.** Having built this, PyTorch's `loss.backward()` /
   `optimizer.step()` will read as familiar machinery rather than magic — which is exactly the
   payoff for writing it yourself once.

---

## Glossary

| Term | Meaning |
|---|---|
| **Unit / neuron** | One output of a layer: weighted sum + bias, then an activation |
| **Weight** | How strongly one input influences one unit |
| **Bias** | A unit's baseline output before inputs are considered |
| **z (pre-activation / logit)** | The weighted sum, before the activation is applied |
| **a (activation)** | `g(z)` — the unit's output |
| **δ (delta)** | `dL/dz` — the gradient after passing back through the activation |
| **Gradient** | Vector of derivatives: which way is uphill, and how steeply |
| **Loss** | One number measuring how wrong the predictions are |
| **Forward pass** | Input → prediction |
| **Backward pass** | Loss → gradients for every parameter |
| **Epoch** | One full pass over the training set |
| **Mini-batch** | A group of examples whose gradients are averaged into one update |
| **Learning rate (η)** | Step size for gradient descent |
| **Fan-in / fan-out** | Number of inputs to / outputs from a layer; sets the init scale |
| **Symmetry breaking** | Random init ensuring units in a layer can learn different features |
| **Linearly separable** | Separable by one straight line/plane (AND yes, XOR no) |
| **Saturation** | An activation pinned at its flat extreme, where the gradient ≈ 0 |
| **Vanishing gradient** | Gradients shrinking toward zero through depth, stalling early layers |
| **Dead unit** | A ReLU unit outputting 0 for every input — gradient permanently 0 |
| **Overfitting** | Memorizing training data instead of learning generalizable patterns |
| **Hyperparameter** | A value you choose rather than learn (learning rate, layer sizes, epochs) |
| **SIMD** | One CPU instruction operating on 4–16 numbers simultaneously |
| **GEMM** | General Matrix Multiply — the batched operation real frameworks are built on |
