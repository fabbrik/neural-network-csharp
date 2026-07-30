using Xunit;

namespace NN.Tests;

public class DenseTests
{
    /// <summary>
    /// The forward pass against hand-computed values: z = 0.3(1.0) + (-0.2)(0.5) + 0.1 = 0.3,
    /// then sigmoid(0.3) = 0.574443. These are the numbers worked through in the study guide.
    /// </summary>
    [Fact]
    public void Forward_matches_a_hand_computed_value()
    {
        var layer = new Dense<Sigmoid>(inputs: 2, units: 1, weights: [0.3f, -0.2f], bias: [0.1f]);

        float a = layer.Forward([1.0f, 0.5f])[0];

        Assert.Equal(0.574443f, a, 5);
    }

    [Fact]
    public void Identity_activation_returns_the_weighted_sum_unchanged()
    {
        var layer = new Dense<Identity>(inputs: 3, units: 1, weights: [1f, 2f, 3f], bias: [0.5f]);

        Assert.Equal(1 * 1f + 2 * 2f + 3 * 3f + 0.5f, layer.Forward([1f, 2f, 3f])[0], 5);
    }

    /// <summary>Exercises the SIMD path plus its scalar tail: 37 is not a multiple of any vector width.</summary>
    [Fact]
    public void Dot_product_is_correct_for_lengths_that_straddle_the_simd_width()
    {
        foreach (int n in new[] { 1, 3, 4, 7, 8, 15, 16, 17, 37, 64 })
        {
            var weights = new float[n];
            var input = new float[n];
            float expected = 0f;

            for (int i = 0; i < n; i++)
            {
                weights[i] = (i % 7) - 3;
                input[i] = (i % 5) * 0.5f;
                expected += weights[i] * input[i];
            }

            var layer = new Dense<Identity>(n, 1, weights, [0f]);

            Assert.Equal(expected, layer.Forward(input)[0], 3);
        }
    }

    [Fact]
    public void ForwardBatch_matches_forwarding_each_example_individually()
    {
        var layer = new Dense<Tanh>(inputs: 2, units: 3);
        layer.Initialize(new Random(42));

        float[] batch = [0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f];
        var actual = new float[4 * 3];

        layer.ForwardBatch(batch, actual, count: 4);

        for (int i = 0; i < 4; i++)
        {
            float[] expected = layer.Forward(batch.AsSpan(i * 2, 2));

            for (int j = 0; j < 3; j++)
                Assert.Equal(expected[j], actual[i * 3 + j]);
        }
    }

    [Fact]
    public void Forward_rejects_a_wrongly_sized_input()
    {
        var layer = new Dense<Tanh>(inputs: 3, units: 2);

        Assert.Throws<ArgumentException>(() => layer.Forward([1f, 2f]));
    }

    [Fact]
    public void Parameter_count_is_weights_plus_biases()
    {
        Assert.Equal(3 * 4 + 4, new Dense<Tanh>(inputs: 3, units: 4).ParameterCount);
    }

    [Fact]
    public void Descriptor_names_the_activation()
    {
        Assert.Equal("Dense<Tanh>", new Dense<Tanh>(2, 2).Descriptor);
        Assert.Equal("Dense<Sigmoid>", new Dense<Sigmoid>(2, 2).Descriptor);
    }

    /// <summary>Xavier initialization must break symmetry, or no unit can specialize.</summary>
    [Fact]
    public void Initialize_produces_distinct_weights()
    {
        var layer = new Dense<Tanh>(inputs: 4, units: 4);
        layer.Initialize(new Random(42));

        Assert.True(layer.Weights.Distinct().Count() > 1, "all weights identical — symmetry unbroken");
        Assert.All(layer.Bias, b => Assert.Equal(0f, b));
    }
}

public class ActivationTests
{
    [Fact]
    public void Sigmoid_is_centred_at_one_half_and_saturates()
    {
        Assert.Equal(0.5f, Sigmoid.Apply(0f), 6);
        Assert.True(Sigmoid.Apply(20f) > 0.99f);
        Assert.True(Sigmoid.Apply(-20f) < 0.01f);
    }

    [Fact]
    public void Tanh_is_zero_centred()
    {
        Assert.Equal(0f, Tanh.Apply(0f), 6);
        Assert.True(Tanh.Apply(-3f) < 0f);
    }

    [Fact]
    public void ReLU_clamps_negatives_to_zero()
    {
        Assert.Equal(0f, ReLU.Apply(-5f));
        Assert.Equal(2.5f, ReLU.Apply(2.5f));
    }

    [Theory]
    [InlineData(0.5f)]
    [InlineData(0.1f)]
    [InlineData(0.9f)]
    public void Sigmoid_derivative_matches_finite_differences(float z)
    {
        float numeric = CentralDifference(Sigmoid.Apply, z);

        AssertClose(numeric, Sigmoid.DerivativeFromOutput(Sigmoid.Apply(z)));
    }

    [Theory]
    [InlineData(0.5f)]
    [InlineData(-0.8f)]
    public void Tanh_derivative_matches_finite_differences(float z)
    {
        float numeric = CentralDifference(Tanh.Apply, z);

        AssertClose(numeric, Tanh.DerivativeFromOutput(Tanh.Apply(z)));
    }

    private static float CentralDifference(Func<float, float> f, float z, float eps = 1e-3f) =>
        (f(z + eps) - f(z - eps)) / (2 * eps);

    /// <summary>
    /// Compares by relative error rather than decimal places. A fixed number of decimals fails
    /// spuriously when the two values straddle a rounding boundary — 0.205487 and 0.205500 differ
    /// by 6e-5 yet round to 0.205 and 0.206.
    /// </summary>
    private static void AssertClose(float expected, float actual, float tolerance = 1e-3f)
    {
        float relative = MathF.Abs(expected - actual) / MathF.Max(1e-6f, MathF.Abs(expected));

        Assert.True(relative < tolerance, $"expected {expected:F6}, got {actual:F6} (relative {relative:E2})");
    }

    /// <summary>
    /// Step must refuse to supply a derivative. Returning its true value (zero) would silently
    /// zero every gradient in the network and training would appear to run while learning nothing.
    /// </summary>
    [Fact]
    public void Step_refuses_to_provide_a_derivative()
    {
        Assert.Throws<NotSupportedException>(() => Step.DerivativeFromOutput(1f));
    }
}
