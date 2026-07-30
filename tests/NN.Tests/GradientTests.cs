using Xunit;

namespace NN.Tests;

/// <summary>
/// Verifies the backward pass against finite differences. These are the most important tests in
/// the suite: a wrong gradient still trains <i>somewhat</i>, so convergence tests alone would not
/// catch it.
/// </summary>
public class GradientTests
{
    private static readonly float[] Input = [0.3f, -0.7f, 0.5f];
    private static readonly float[] Target = [1f, 0f];

    [Fact]
    public void Backward_matches_finite_differences_for_tanh_and_sigmoid()
    {
        var net = new Sequential(inputs: 3)
            .Dense<Tanh>(4)
            .Dense<Sigmoid>(2)
            .Build(seed: 7);

        float error = GradientCheck.MaxRelativeError(net, Input, Target);

        Assert.True(error < 1e-3f, $"max relative error {error:E3} exceeds 1e-3");
    }

    [Fact]
    public void Backward_matches_finite_differences_for_relu_and_identity()
    {
        var net = new Sequential(inputs: 3)
            .Dense<ReLU>(5)
            .Dense<Identity>(2)
            .Build(seed: 11);

        float error = GradientCheck.MaxRelativeError(net, Input, Target);

        Assert.True(error < 1e-3f, $"max relative error {error:E3} exceeds 1e-3");
    }

    /// <summary>
    /// Deeper networks tolerate less: each extra layer compounds float32 roundoff in the loss
    /// evaluations the finite difference relies on, raising the achievable error floor (measured
    /// ~4.6e-3 here versus ~2.4e-4 for two layers). The threshold is loosened accordingly — it
    /// still separates a correct gradient from a broken one by more than an order of magnitude,
    /// as <see cref="Gradient_check_detects_a_wrong_derivative"/> shows at &gt;1e-1.
    /// </summary>
    [Fact]
    public void Backward_matches_finite_differences_through_three_layers()
    {
        var net = new Sequential(inputs: 3)
            .Dense<Tanh>(6)
            .Dense<Tanh>(4)
            .Dense<Sigmoid>(2)
            .Build(seed: 3);

        float error = GradientCheck.MaxRelativeError(net, Input, Target);

        Assert.True(error < 1e-2f, $"max relative error {error:E3} exceeds 1e-2");
    }

    /// <summary>
    /// The accuracy U-curve documented in the study guide: error is worse at both large ε
    /// (truncation) and small ε (float32 roundoff), with a minimum in between. A wrong gradient
    /// would show O(1) error at every ε, so the presence of the trough is itself evidence of
    /// correctness.
    /// </summary>
    [Fact]
    public void Finite_difference_accuracy_has_a_sweet_spot_at_the_documented_epsilon()
    {
        var net = new Sequential(inputs: 3).Dense<Tanh>(4).Dense<Sigmoid>(2).Build(seed: 7);

        float coarse = GradientCheck.MaxRelativeError(net, Input, Target, epsilon: 1e-1f);
        float best = GradientCheck.MaxRelativeError(net, Input, Target, epsilon: 1e-2f);
        float tooFine = GradientCheck.MaxRelativeError(net, Input, Target, epsilon: 1e-4f);

        Assert.True(best < coarse, $"expected 1e-2 ({best:E3}) to beat 1e-1 ({coarse:E3})");
        Assert.True(best < tooFine, $"expected 1e-2 ({best:E3}) to beat 1e-4 ({tooFine:E3})");
    }

    /// <summary>
    /// A deliberately corrupted derivative must be caught. This is the test that proves the other
    /// gradient tests can actually fail — without it they might pass vacuously.
    /// </summary>
    [Fact]
    public void Gradient_check_detects_a_wrong_derivative()
    {
        var net = new Sequential(inputs: 3).Dense<BrokenTanh>(4).Dense<Sigmoid>(2).Build(seed: 7);

        float error = GradientCheck.MaxRelativeError(net, Input, Target);

        Assert.True(error > 0.1f, $"a broken derivative should produce O(1) error, got {error:E3}");
    }

    /// <summary>Tanh with the sign of its derivative flipped: <c>1 + a²</c> instead of <c>1 - a²</c>.</summary>
    private readonly struct BrokenTanh : IActivation
    {
        public static float Apply(float z) => MathF.Tanh(z);
        public static float DerivativeFromOutput(float a) => 1f + a * a;
    }
}
