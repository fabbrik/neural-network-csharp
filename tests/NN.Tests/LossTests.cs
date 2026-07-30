using Xunit;

namespace NN.Tests;

/// <summary>
/// Softmax as a function: the properties everything downstream relies on.
/// </summary>
public class SoftmaxTests
{
    private static float[] Softmax(params float[] logits)
    {
        var copy = (float[])logits.Clone();
        SoftmaxCrossEntropy.Instance.Transform(copy);
        return copy;
    }

    [Fact]
    public void Produces_a_probability_distribution()
    {
        float[] p = Softmax(1f, 2f, 3f, -1f);

        Assert.All(p, v => Assert.InRange(v, 0f, 1f));
        Assert.Equal(1f, p.Sum(), 5);
    }

    [Fact]
    public void Preserves_the_ordering_of_the_logits()
    {
        float[] p = Softmax(0.5f, 3f, -2f, 1f);

        Assert.Equal(1, Array.IndexOf(p, p.Max()));
        Assert.Equal(2, Array.IndexOf(p, p.Min()));
    }

    [Fact]
    public void Equal_logits_give_a_uniform_distribution()
    {
        float[] p = Softmax(4f, 4f, 4f, 4f);

        Assert.All(p, v => Assert.Equal(0.25f, v, 5));
    }

    /// <summary>
    /// Softmax is shift-invariant: adding a constant to every logit changes nothing, because the
    /// constant appears in every numerator and the denominator. This is exactly the property the
    /// implementation exploits to avoid overflow.
    /// </summary>
    [Fact]
    public void Is_invariant_to_adding_a_constant_to_every_logit()
    {
        float[] a = Softmax(1f, 2f, 3f);
        float[] b = Softmax(101f, 102f, 103f);

        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i], 5);
    }

    /// <summary>
    /// The naive formula computes exp(z) directly, and exp overflows to infinity above about 88
    /// in float32 — giving inf/inf = NaN. Logits this large are ordinary in a trained network,
    /// so subtracting the maximum first is a correctness requirement, not a refinement.
    /// </summary>
    [Fact]
    public void Does_not_overflow_on_large_logits()
    {
        float[] p = Softmax(1000f, 999f, 998f);

        Assert.All(p, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v), $"got {v}"));
        Assert.Equal(1f, p.Sum(), 5);
        Assert.True(p[0] > p[1] && p[1] > p[2]);
    }

    [Fact]
    public void Does_not_underflow_to_nonsense_on_very_negative_logits()
    {
        float[] p = Softmax(-1000f, -999f, 0f);

        Assert.All(p, v => Assert.False(float.IsNaN(v)));
        Assert.Equal(1f, p.Sum(), 5);
        Assert.Equal(1f, p[2], 4);
    }
}

public class CrossEntropyTests
{
    private static float Loss(float[] probabilities, float[] targets) =>
        SoftmaxCrossEntropy.Instance.Evaluate(probabilities, targets);

    /// <summary>Cross-entropy is the negative log-probability given to the true class.</summary>
    [Fact]
    public void Scores_the_probability_assigned_to_the_correct_class()
    {
        Assert.Equal(-MathF.Log(0.7f), Loss([0.2f, 0.7f, 0.1f], [0f, 1f, 0f]), 5);
    }

    [Fact]
    public void A_perfect_confident_prediction_costs_almost_nothing()
    {
        Assert.Equal(0f, Loss([0f, 1f, 0f], [0f, 1f, 0f]), 5);
    }

    /// <summary>
    /// The property that makes it better than MSE for classification: being confidently wrong is
    /// punished without bound, where MSE caps the penalty at 1 per output.
    /// </summary>
    [Fact]
    public void Being_confidently_wrong_is_punished_far_harder_than_being_unsure()
    {
        float unsure = Loss([0.34f, 0.33f, 0.33f], [0f, 1f, 0f]);
        float wrong = Loss([0.99f, 0.005f, 0.005f], [0f, 1f, 0f]);

        Assert.True(wrong > unsure * 4, $"confidently wrong ({wrong:F2}) should dwarf unsure ({unsure:F2})");
    }

    /// <summary>log(0) is negative infinity; the floor keeps a zeroed-out class finite.</summary>
    [Fact]
    public void A_zero_probability_on_the_true_class_stays_finite()
    {
        float loss = Loss([1f, 0f, 0f], [0f, 1f, 0f]);

        Assert.False(float.IsInfinity(loss) || float.IsNaN(loss));
        Assert.True(loss > 10f, "it should still be an enormous penalty");
    }

    /// <summary>The fused gradient is exactly prediction minus target.</summary>
    [Fact]
    public void The_gradient_is_prediction_minus_target()
    {
        var into = new float[3];
        SoftmaxCrossEntropy.Instance.Gradient([0.2f, 0.7f, 0.1f], [0f, 1f, 0f], into);

        Assert.Equal(0.2f, into[0], 5);
        Assert.Equal(-0.3f, into[1], 5);
        Assert.Equal(0.1f, into[2], 5);
    }
}

/// <summary>
/// The tests that actually matter: the fused <c>p - y</c> gradient must agree with finite
/// differences of the real loss.
///
/// <para>The fusion is an algebraic shortcut — softmax's Jacobian and cross-entropy's reciprocal
/// cancelling — and a shortcut that is <i>almost</i> right is the worst possible outcome, because
/// the network still trains. §21's whole argument applies with extra force here.</para>
/// </summary>
public class SoftmaxGradientTests
{
    private static readonly float[] Input = [0.3f, -0.7f, 0.5f];

    /// <summary>One-hot target, as a classifier's would be.</summary>
    private static readonly float[] Target = [0f, 1f, 0f, 0f];

    [Fact]
    public void The_fused_gradient_matches_finite_differences()
    {
        var net = new Sequential(inputs: 3)
            .Dense<Tanh>(5)
            .SoftmaxOutput(4)
            .Build(seed: 7);

        float error = GradientCheck.MaxRelativeError(net, Input, Target);

        Assert.True(error < 1e-3f, $"max relative error {error:E3} exceeds 1e-3");
    }

    [Fact]
    public void The_fused_gradient_matches_finite_differences_through_three_layers()
    {
        var net = new Sequential(inputs: 3)
            .Dense<Tanh>(6)
            .Dense<Tanh>(4)
            .SoftmaxOutput(4)
            .Build(seed: 3);

        float error = GradientCheck.MaxRelativeError(net, Input, Target);

        Assert.True(error < 1e-2f, $"max relative error {error:E3} exceeds 1e-2");
    }

    /// <summary>
    /// The same U-curve §21 uses as evidence for MSE. A wrong gradient would be O(1) at every ε;
    /// a correct one has a sweet spot where truncation and roundoff trade off.
    /// </summary>
    [Fact]
    public void Finite_difference_accuracy_has_a_sweet_spot()
    {
        var net = new Sequential(inputs: 3).Dense<Tanh>(5).SoftmaxOutput(4).Build(seed: 7);

        float coarse = GradientCheck.MaxRelativeError(net, Input, Target, epsilon: 1e-1f);
        float best = GradientCheck.MaxRelativeError(net, Input, Target, epsilon: 1e-2f);

        Assert.True(best < coarse, $"expected 1e-2 ({best:E3}) to beat 1e-1 ({coarse:E3})");
    }

    /// <summary>
    /// Pairing softmax cross-entropy with a squashed output layer must be refused. The fused
    /// gradient assumes raw logits, so a sigmoid output would make it quietly wrong — the network
    /// would train, badly, with nothing to indicate why.
    /// </summary>
    [Fact]
    public void A_squashed_output_layer_is_rejected_rather_than_silently_miscomputed()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new Sequential(inputs: 3)
                .Dense<Tanh>(5)
                .Dense<Sigmoid>(4)
                .WithLoss(SoftmaxCrossEntropy.Instance)
                .Build(seed: 7));

        Assert.Contains("Dense<Sigmoid>", ex.Message);
        Assert.Contains("logits", ex.Message);
    }

    /// <summary>Predict must return probabilities, not the logits the last layer produced.</summary>
    [Fact]
    public void Predict_returns_probabilities()
    {
        var net = new Sequential(inputs: 3).Dense<Tanh>(5).SoftmaxOutput(4).Build(seed: 7);

        ReadOnlySpan<float> output = net.Predict(Input);

        float sum = 0f;
        foreach (float v in output) sum += v;

        Assert.Equal(1f, sum, 5);
    }

    /// <summary>
    /// The point of the whole exercise: a softmax classifier should learn a classification task,
    /// and do it without the enormous learning rate MSE-over-sigmoid demands.
    /// </summary>
    [Fact]
    public void A_softmax_classifier_learns_a_four_way_task_at_a_normal_learning_rate()
    {
        // Four inputs, each belonging to its own class — a trivial task, but one that requires
        // the gradient to point the right way for every class.
        float[] x = [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1];
        float[] y = [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1];

        var net = new Sequential(inputs: 4).Dense<Tanh>(8).SoftmaxOutput(4).Build(seed: 42);

        net.Train(x, y, epochs: 500, learningRate: 0.1f);

        for (int i = 0; i < 4; i++)
        {
            ReadOnlySpan<float> p = net.Predict(x.AsSpan(i * 4, 4));

            int best = 0;
            for (int d = 1; d < 4; d++) if (p[d] > p[best]) best = d;

            Assert.Equal(i, best);
            Assert.True(p[best] > 0.8f, $"class {i} predicted at only {p[best]:F2}");
        }
    }
}

/// <summary>The loss travels with the model, because a classifier's weights are useless without it.</summary>
public class LossPersistenceTests
{
    [Fact]
    public void A_softmax_model_reloads_as_a_softmax_model()
    {
        var net = new Sequential(inputs: 3).Dense<Tanh>(5).SoftmaxOutput(4).Build(seed: 7);

        using var stream = new MemoryStream();
        ModelIO.Save(net, stream);
        stream.Position = 0;
        Network loaded = ModelIO.Load(stream);

        Assert.Equal(net.LossFunction.Name, loaded.LossFunction.Name);
        Assert.Equal("softmax-cross-entropy", loaded.LossFunction.Name);

        // The observable consequence: outputs are still probabilities, not logits.
        float sum = 0f;
        foreach (float v in loaded.Predict([0.3f, -0.7f, 0.5f])) sum += v;

        Assert.Equal(1f, sum, 5);
    }

    [Fact]
    public void An_mse_model_reloads_as_an_mse_model()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);

        using var stream = new MemoryStream();
        ModelIO.Save(net, stream);
        stream.Position = 0;

        Assert.Equal("mse", ModelIO.Load(stream).LossFunction.Name);
    }

    /// <summary>
    /// Version 1 files predate the loss field and must still load, as mean squared error — which
    /// is what they were. The repository ships one such model, so this is not hypothetical.
    /// </summary>
    [Fact]
    public void A_version_1_file_still_loads_as_mean_squared_error()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);

        using var current = new MemoryStream();
        ModelIO.Save(net, current);

        // Rewrite the stream as a version 1 file: same bytes, but version 1 and no loss string.
        byte[] v2 = current.ToArray();
        using var legacy = new MemoryStream();
        using (var writer = new BinaryWriter(legacy, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(v2.AsSpan(0, 8).ToArray());   // magic
            writer.Write(1);                            // version 1

            // Skip the version-2 loss string: one length byte for a short name, then its bytes.
            int lossLength = v2[12];
            writer.Write(v2.AsSpan(13 + lossLength).ToArray());
        }

        legacy.Position = 0;
        Network loaded = ModelIO.Load(legacy);

        Assert.Equal("mse", loaded.LossFunction.Name);
        Assert.Equal(net.ParameterCount, loaded.ParameterCount);
    }

    [Fact]
    public void An_unknown_loss_name_is_reported_clearly()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);

        using var stream = new MemoryStream();
        ModelIO.Save(net, stream);

        byte[] bytes = stream.ToArray();
        // "mse" is three characters; overwrite them with a name nothing registers.
        bytes[13] = (byte)'x';
        bytes[14] = (byte)'y';
        bytes[15] = (byte)'z';

        var ex = Assert.Throws<InvalidDataException>(() => ModelIO.Load(new MemoryStream(bytes)));

        Assert.Contains("xyz", ex.Message);
    }
}
