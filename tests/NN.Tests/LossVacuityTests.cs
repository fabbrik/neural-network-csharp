using Xunit;

namespace NN.Tests;

/// <summary>
/// Proof that the softmax cross-entropy gradient tests in <c>LossTests</c> can actually fail.
///
/// <para>This is the same argument §21 makes about the MSE gradient check, applied to the fused
/// output gradient. Tests asserting "the numbers agree" are only evidence if disagreement is
/// reachable — otherwise they might be passing vacuously, and a check that cannot fail proves
/// nothing. So here the derivative is deliberately corrupted in the two most plausible ways and
/// the check is required to catch both.</para>
/// </summary>
public class LossVacuityTests
{
    private static readonly float[] Input = [0.3f, -0.7f, 0.5f];
    private static readonly float[] Target = [0f, 1f, 0f, 0f];

    private static float CheckWith(ILoss loss) =>
        GradientCheck.MaxRelativeError(
            new Sequential(inputs: 3).Dense<Tanh>(6).Dense<Identity>(4).WithLoss(loss).Build(seed: 7),
            Input, Target);

    /// <summary>
    /// The correct fused gradient, as a control: whatever the corrupted versions score, this is
    /// the number they have to be worse than for the comparison to mean anything.
    /// </summary>
    [Fact]
    public void The_correct_fused_gradient_passes()
    {
        Assert.True(CheckWith(SoftmaxCrossEntropy.Instance) < 1e-3f);
    }

    /// <summary>
    /// Sign flip: <c>y - p</c> instead of <c>p - y</c>. Easy to write, and catastrophic — it
    /// performs gradient *ascent*, climbing the loss instead of descending it.
    /// </summary>
    [Fact]
    public void A_sign_flipped_gradient_is_caught()
    {
        float error = CheckWith(new CorruptedCrossEntropy(flipSign: true, skipSoftmax: false));

        Assert.True(error > 0.1f, $"a sign-flipped gradient should show O(1) error, got {error:E3}");
    }

    /// <summary>
    /// The subtler mistake: taking <c>p - y</c> against the raw logits rather than the softmax
    /// probabilities. The shapes match, nothing throws, and training still partly works — which
    /// is exactly the class of bug finite differences exist to find.
    /// </summary>
    [Fact]
    public void Using_logits_instead_of_probabilities_is_caught()
    {
        float error = CheckWith(new CorruptedCrossEntropy(flipSign: false, skipSoftmax: true));

        Assert.True(error > 0.1f, $"differentiating against logits should show O(1) error, got {error:E3}");
    }

    /// <summary>Softmax cross-entropy with a deliberately broken backward pass.</summary>
    private sealed class CorruptedCrossEntropy(bool flipSign, bool skipSoftmax) : ILoss
    {
        public string Name => "corrupted-for-testing";

        public void Transform(Span<float> outputs) => SoftmaxCrossEntropy.Instance.Transform(outputs);

        public float Evaluate(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets) =>
            SoftmaxCrossEntropy.Instance.Evaluate(outputs, targets);

        public void Gradient(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets, Span<float> into)
        {
            for (int j = 0; j < outputs.Length; j++)
            {
                // "Undo" the softmax badly, to stand in for having differentiated the logits.
                float p = skipSoftmax ? MathF.Log(MathF.Max(outputs[j], 1e-7f)) : outputs[j];

                into[j] = flipSign ? targets[j] - p : p - targets[j];
            }
        }

        public void Validate(ILayer outputLayer) { }
    }
}

/// <summary>The loss a network gets when nobody asks for one.</summary>
public class LossDefaultTests
{
    [Fact]
    public void A_network_defaults_to_mean_squared_error()
    {
        var built = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 1);
        var direct = new Network(42, new Dense<Tanh>(2, 4), new Dense<Sigmoid>(4, 1));

        Assert.IsType<MeanSquaredError>(built.LossFunction);
        Assert.IsType<MeanSquaredError>(direct.LossFunction);
    }

    /// <summary>
    /// MSE must leave the output alone. If it ever normalized, every regression network in the
    /// repo would silently start predicting something else.
    /// </summary>
    [Fact]
    public void Mean_squared_error_does_not_transform_the_output()
    {
        float[] outputs = [3f, -1f, 0.5f];
        var copy = (float[])outputs.Clone();

        MeanSquaredError.Instance.Transform(copy);

        Assert.Equal(outputs, copy);
    }

    /// <summary>A perfect prediction leaves nothing to learn from.</summary>
    [Fact]
    public void A_perfect_softmax_prediction_produces_a_zero_gradient()
    {
        var gradient = new float[3];

        SoftmaxCrossEntropy.Instance.Gradient([1f, 0f, 0f], [1f, 0f, 0f], gradient);

        Assert.All(gradient, g => Assert.Equal(0f, g, 6));
    }
}
