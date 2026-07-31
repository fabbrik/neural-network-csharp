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
        public static void ApplyAll(Span<float> z) => IActivation.Elementwise<BrokenTanh>(z);
        public static float DerivativeFromOutput(float a) => 1f + a * a;
    }
}

/// <summary>
/// The one case where a large gradient-check error does <i>not</i> mean a broken backward pass.
///
/// <para>Finite differences assume the loss is smooth across [w−ε, w+ε]. ReLU has a kink at
/// z = 0, so if a perturbation of ε pushes some unit's z across zero, the two loss evaluations
/// straddle the corner and their difference measures the average of two different slopes. The
/// analytic gradient is correct; the <i>numerical estimate</i> is not.</para>
///
/// <para>These tests pin the distinction the study guide (§21) draws between a kink artifact and
/// a real bug: a real bug fails at every ε and for every activation, and a kink artifact does
/// neither. Without them, a reader hitting this would have no way to tell which they were
/// looking at — and the natural conclusion is that the library is broken.</para>
/// </summary>
public class ReLUKinkTests
{
    private static readonly float[] Input = [1f, 1f];
    private static readonly float[] Target = [1f];

    /// <summary>
    /// One ReLU unit feeding an identity output, with parameters chosen so that
    /// <c>z = w₀ + w₁ + b</c> sits <paramref name="distanceToKink"/> above zero. A perturbation
    /// larger than that distance drags z across the corner.
    /// </summary>
    private static Network OneReluUnit(float distanceToKink)
    {
        var net = new Network(1, new Dense<ReLU>(2, 1), new Dense<Identity>(1, 1));

        // w₀ = w₁ = 1 and x = [1, 1], so b positions z exactly.
        float[] hidden = [1f, 1f, distanceToKink - 2f];
        for (int p = 0; p < hidden.Length; p++) net.Layers[0].SetParameter(p, hidden[p]);

        net.Layers[1].SetParameter(0, 1f);   // weight
        net.Layers[1].SetParameter(1, 0f);   // bias

        return net;
    }

    /// <summary>Tanh in the identical shape — same weights, same arithmetic, no corner.</summary>
    private static Network OneTanhUnit(float z)
    {
        var net = new Network(1, new Dense<Tanh>(2, 1), new Dense<Identity>(1, 1));

        float[] hidden = [1f, 1f, z - 2f];
        for (int p = 0; p < hidden.Length; p++) net.Layers[0].SetParameter(p, hidden[p]);

        net.Layers[1].SetParameter(0, 1f);
        net.Layers[1].SetParameter(1, 0f);

        return net;
    }

    /// <summary>
    /// z sits 0.001 from the kink and the default ε is 0.01, so the check straddles the corner
    /// and reports an error that would ordinarily condemn the backward pass.
    /// </summary>
    [Fact]
    public void A_perturbation_across_the_kink_reports_a_large_error_despite_a_correct_gradient()
    {
        float error = GradientCheck.MaxRelativeError(OneReluUnit(0.001f), Input, Target);

        Assert.True(error > 0.1f,
            $"expected the kink artifact to look like a failure, got {error:E3}");
    }

    /// <summary>
    /// The diagnostic that separates artifact from bug: shrink ε below the distance to the kink
    /// and the error collapses. A genuinely wrong derivative stays O(1) at every ε — compare
    /// <see cref="GradientTests.Gradient_check_detects_a_wrong_derivative"/>.
    /// </summary>
    [Fact]
    public void Shrinking_epsilon_below_the_distance_to_the_kink_recovers_agreement()
    {
        Network net = OneReluUnit(0.001f);

        float straddling = GradientCheck.MaxRelativeError(net, Input, Target, epsilon: 1e-2f);
        float clear = GradientCheck.MaxRelativeError(net, Input, Target, epsilon: 1e-4f);

        Assert.True(clear < straddling / 3f,
            $"a smaller ε should escape the corner: {straddling:E3} -> {clear:E3}");
    }

    /// <summary>The second diagnostic: the same shape with a smooth activation passes cleanly.</summary>
    [Fact]
    public void The_same_network_shape_with_tanh_shows_no_such_error()
    {
        float error = GradientCheck.MaxRelativeError(OneTanhUnit(0.001f), Input, Target);

        Assert.True(error < 1e-3f, $"tanh has no corner to straddle, but got {error:E3}");
    }

    /// <summary>And a ReLU unit comfortably clear of the kink checks out at the default ε.</summary>
    [Fact]
    public void A_relu_unit_far_from_the_kink_passes_normally()
    {
        float error = GradientCheck.MaxRelativeError(OneReluUnit(1f), Input, Target);

        Assert.True(error < 1e-3f, $"max relative error {error:E3} exceeds 1e-3");
    }
}
