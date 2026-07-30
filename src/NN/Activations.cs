using System.Runtime.CompilerServices;

namespace NN;

/// <summary>
/// Activation function. Declared with static abstract members so the JIT devirtualizes
/// and inlines the call at each generic instantiation — no delegate dispatch in the inner loop.
/// </summary>
public interface IActivation
{
    /// <summary>Maps the pre-activation <paramref name="z"/> to the unit's output.</summary>
    static abstract float Apply(float z);

    /// <summary>
    /// Derivative g'(z), expressed in terms of the already-computed output <c>a = g(z)</c>.
    /// Every activation here has a derivative cheaply recoverable from its output, so backprop
    /// reuses the cached forward activations instead of re-deriving z.
    /// </summary>
    static abstract float DerivativeFromOutput(float a);
}

public readonly struct Sigmoid : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => 1f / (1f + MathF.Exp(-z));

    // σ'(z) = σ(z)(1 - σ(z))
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => a * (1f - a);
}

public readonly struct Tanh : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => MathF.Tanh(z);

    // tanh'(z) = 1 - tanh²(z)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => 1f - a * a;
}

public readonly struct ReLU : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => z > 0f ? z : 0f;

    // Undefined at 0; the usual convention picks 0.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => a > 0f ? 1f : 0f;
}

public readonly struct Step : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => z >= 0f ? 1f : 0f;

    /// <summary>
    /// The step function has a zero gradient everywhere it is defined, which is precisely why
    /// backprop cannot train it and why the perceptron needs its own update rule. Throws rather
    /// than silently zeroing every gradient in the network.
    /// </summary>
    public static float DerivativeFromOutput(float a) =>
        throw new NotSupportedException("Step is not differentiable — use Sigmoid, Tanh, or ReLU for backprop.");
}

public readonly struct Identity : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => z;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => 1f;
}
