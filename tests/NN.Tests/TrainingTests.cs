using Xunit;

namespace NN.Tests;

public class TrainingTests
{
    private static readonly float[] LogicInputs = [0, 0, 0, 1, 1, 0, 1, 1];
    private static readonly float[] AndTargets = [0, 0, 0, 1];
    private static readonly float[] XorTargets = [0, 1, 1, 0];

    [Fact]
    public void Perceptron_converges_on_linearly_separable_data()
    {
        var p = new Perceptron(inputs: 2);

        int epochs = p.Train(LogicInputs, AndTargets, epochs: 100, learningRate: 0.1f);

        Assert.True(epochs < 100, "should converge and stop early, not exhaust the epoch budget");

        for (int i = 0; i < AndTargets.Length; i++)
            Assert.Equal(AndTargets[i], p.Predict(LogicInputs.AsSpan(i * 2, 2)));
    }

    /// <summary>
    /// The Minsky–Papert result: a single perceptron cannot represent XOR, so it never converges
    /// and burns every epoch. This is a property of the algorithm, not a bug.
    /// </summary>
    [Fact]
    public void Perceptron_cannot_learn_xor()
    {
        var p = new Perceptron(inputs: 2);

        int epochs = p.Train(LogicInputs, XorTargets, epochs: 100, learningRate: 0.1f);

        Assert.Equal(100, epochs);

        bool allCorrect = true;
        for (int i = 0; i < XorTargets.Length; i++)
            if (p.Predict(LogicInputs.AsSpan(i * 2, 2)) != XorTargets[i]) allCorrect = false;

        Assert.False(allCorrect, "a single perceptron must not be able to solve XOR");
    }

    /// <summary>What the hidden layer buys: the same problem the perceptron above cannot solve.</summary>
    [Fact]
    public void Network_with_hidden_layer_learns_xor()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);

        float loss = net.Train(LogicInputs, XorTargets, epochs: 4000, learningRate: 0.5f);

        Assert.True(loss < 0.01f, $"expected loss below 0.01, got {loss:F6}");

        for (int i = 0; i < XorTargets.Length; i++)
        {
            float predicted = net.Predict(LogicInputs.AsSpan(i * 2, 2))[0];
            Assert.Equal(XorTargets[i], MathF.Round(predicted));
        }
    }

    /// <summary>
    /// Zero-initialized weights kill the gradient before it reaches the hidden layer, so nothing
    /// learns at all — the network sits at 0.25 loss predicting 0.5 forever. Guards the claim the
    /// study guide makes about why initialization matters.
    /// </summary>
    [Fact]
    public void Zero_initialized_weights_cannot_learn()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);

        foreach (ILayer layer in net.Layers)
            for (int p = 0; p < layer.ParameterCount; p++)
                layer.SetParameter(p, 0f);

        float loss = net.Train(LogicInputs, XorTargets, epochs: 500, learningRate: 0.5f);

        Assert.Equal(0.25f, loss, 5);
        Assert.Equal(0.5f, net.Predict(LogicInputs.AsSpan(0, 2))[0], 5);
    }

    [Fact]
    public void Training_is_reproducible_for_a_fixed_seed()
    {
        static float Run() =>
            new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42)
                .Train(LogicInputs, XorTargets, epochs: 200, learningRate: 0.5f);

        Assert.Equal(Run(), Run());
    }

    [Fact]
    public void Different_seeds_produce_different_solutions()
    {
        var a = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 1);
        var b = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 2);

        Assert.NotEqual(a.Layers[0].GetParameter(0), b.Layers[0].GetParameter(0));
    }

    [Fact]
    public void Mini_batch_training_reduces_loss()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 5);

        float before = net.Loss(LogicInputs.AsSpan(0, 2), XorTargets.AsSpan(0, 1));
        net.Train(LogicInputs, XorTargets, epochs: 2000, learningRate: 0.5f, batchSize: 2);
        float after = net.Train(LogicInputs, XorTargets, epochs: 1, learningRate: 0f, batchSize: 2);

        Assert.True(after < before, $"loss should fall: {before:F6} -> {after:F6}");
    }
}
