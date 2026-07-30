using Xunit;

namespace NN.Tests;

public class MoonsTests
{
    [Fact]
    public void Produces_row_major_arrays_sized_for_two_input_features()
    {
        (float[] x, float[] y) = Datasets.Moons(count: 100);

        Assert.Equal(200, x.Length);
        Assert.Equal(100, y.Length);
    }

    /// <summary>
    /// The demo splits the data positionally. That is only legitimate because classes alternate,
    /// so every contiguous slice is balanced — otherwise the "test set" could be one whole class.
    /// </summary>
    [Fact]
    public void Classes_alternate_so_any_positional_split_stays_balanced()
    {
        (_, float[] y) = Datasets.Moons(count: 100);

        Assert.All(y, label => Assert.True(label is 0f or 1f));

        foreach (int split in new[] { 10, 50, 90 })
        {
            Assert.Equal(split / 2, y[..split].Count(l => l == 1f));
            Assert.Equal((100 - split) / 2, y[split..].Count(l => l == 1f));
        }
    }

    [Fact]
    public void The_same_seed_gives_the_same_dataset_and_different_seeds_do_not()
    {
        (float[] a, _) = Datasets.Moons(count: 50, seed: 3);
        (float[] b, _) = Datasets.Moons(count: 50, seed: 3);
        (float[] c, _) = Datasets.Moons(count: 50, seed: 4);

        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
    }

    /// <summary>Without noise the moons are exact half-circles, so the geometry is checkable.</summary>
    [Fact]
    public void With_zero_noise_each_point_lies_on_its_moons_arc()
    {
        (float[] x, float[] y) = Datasets.Moons(count: 200, noise: 0f);

        for (int i = 0; i < y.Length; i++)
        {
            // Upper moon is the unit half-circle at the origin; the lower one is the same arc
            // flipped and shifted to (1, 0.5), which is what makes them interlock.
            float cx = y[i] == 1f ? 0f : 1f;
            float cy = y[i] == 1f ? 0f : 0.5f;

            float dx = x[i * 2] - cx, dy = x[i * 2 + 1] - cy;

            Assert.Equal(1f, MathF.Sqrt(dx * dx + dy * dy), 4);
        }
    }

    [Fact]
    public void Noise_widens_the_spread_around_the_arcs()
    {
        (float[] clean, _) = Datasets.Moons(count: 400, noise: 0f, seed: 1);
        (float[] noisy, _) = Datasets.Moons(count: 400, noise: 0.3f, seed: 1);

        Assert.True(Spread(noisy) > Spread(clean), "noise should widen the point cloud");

        static float Spread(float[] points)
        {
            float mean = points.Average();
            return points.Sum(v => (v - mean) * (v - mean)) / points.Length;
        }
    }

    [Fact]
    public void Invalid_sizes_and_negative_noise_are_rejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Datasets.Moons(count: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Datasets.Moons(count: 10, noise: -0.1f));
    }
}

/// <summary>
/// The properties XOR cannot demonstrate, on a dataset that can: generalization to held-out data,
/// and the fact that the problem needs a hidden layer at all.
/// </summary>
public class GeneralizationTests
{
    private static (float[] TrainX, float[] TrainY, float[] TestX, float[] TestY) Split()
    {
        (float[] x, float[] y) = Datasets.Moons(count: 1500, noise: 0.2f, seed: 0);

        return (x[..2000], y[..1000], x[2000..], y[1000..]);
    }

    private static float Accuracy(Network net, float[] x, float[] y)
    {
        int correct = 0;

        for (int i = 0; i < y.Length; i++)
            if (MathF.Round(net.Predict(x.AsSpan(i * 2, 2))[0]) == y[i]) correct++;

        return (float)correct / y.Length;
    }

    /// <summary>
    /// Mini-batch training on data the network has never seen must generalize, not memorize.
    /// This is the test XOR structurally cannot provide: with four examples and no held-out set,
    /// "learned it" and "memorized it" are the same measurement.
    /// </summary>
    [Fact]
    public void A_small_network_generalizes_to_held_out_moons()
    {
        (float[] trainX, float[] trainY, float[] testX, float[] testY) = Split();

        var net = new Sequential(inputs: 2)
            .Dense<Tanh>(16)
            .Dense<Tanh>(16)
            .Dense<Sigmoid>(1)
            .Build(seed: 7);

        net.Train(trainX, trainY, epochs: 150, learningRate: 0.3f, batchSize: 32);

        float test = Accuracy(net, testX, testY);

        Assert.True(test > 0.9f, $"expected better than 90% on held-out data, got {test:P1}");
        Assert.True(MathF.Abs(Accuracy(net, trainX, trainY) - test) < 0.1f,
            "train and test accuracy should be close for a well-sized network");
    }

    /// <summary>
    /// The moons are not linearly separable, so a single-layer network — no hidden layer, hence
    /// only a straight boundary — must do measurably worse. Same role AND-vs-XOR plays for the
    /// perceptron, on a problem with enough data for the gap to be a statistic rather than a
    /// count of four points.
    /// </summary>
    [Fact]
    public void A_network_with_no_hidden_layer_does_measurably_worse()
    {
        (float[] trainX, float[] trainY, float[] testX, float[] testY) = Split();

        var linear = new Sequential(inputs: 2).Dense<Sigmoid>(1).Build(seed: 7);
        linear.Train(trainX, trainY, epochs: 150, learningRate: 0.3f, batchSize: 32);

        var hidden = new Sequential(inputs: 2)
            .Dense<Tanh>(16)
            .Dense<Tanh>(16)
            .Dense<Sigmoid>(1)
            .Build(seed: 7);
        hidden.Train(trainX, trainY, epochs: 150, learningRate: 0.3f, batchSize: 32);

        float straight = Accuracy(linear, testX, testY);
        float curved = Accuracy(hidden, testX, testY);

        Assert.True(curved > straight + 0.05f,
            $"the hidden layer should buy real accuracy: {straight:P1} -> {curved:P1}");
    }

    /// <summary>
    /// Overfitting, as an assertion: far too much capacity on far too little data reaches perfect
    /// training accuracy and still loses on held-out data to a smaller network given more of it.
    /// </summary>
    [Fact]
    public void Too_much_capacity_on_too_little_data_memorizes_instead_of_generalizing()
    {
        (float[] trainX, float[] trainY, float[] testX, float[] testY) = Split();

        var overfit = new Sequential(inputs: 2)
            .Dense<Tanh>(64)
            .Dense<Tanh>(64)
            .Dense<Sigmoid>(1)
            .Build(seed: 7);

        float[] tinyX = trainX[..40], tinyY = trainY[..20];
        overfit.Train(tinyX, tinyY, epochs: 3000, learningRate: 0.5f, batchSize: 8);

        Assert.Equal(1f, Accuracy(overfit, tinyX, tinyY));
        Assert.True(Accuracy(overfit, testX, testY) < 0.95f,
            "a memorizing network should not match a well-trained one on held-out data");
    }
}
