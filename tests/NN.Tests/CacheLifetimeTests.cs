using Xunit;

namespace NN.Tests;

/// <summary>
/// The contract between <see cref="ILayer.ForwardTrain"/> and <see cref="ILayer.Backward"/>.
///
/// <para>These guard a bug class rather than a bug: when one method both computed activations and
/// cached them for backprop, any incidental forward pass — evaluating a loss, logging a prediction
/// — silently rewrote the cache, and the next backward pass computed gradients for one example
/// using another example's activations. Nothing threw. Loss still fell. Splitting the two entry
/// points makes it unrepresentable, and these tests hold that split in place.</para>
/// </summary>
public class CacheLifetimeTests
{
    private static readonly float[] Input = [0.3f, -0.7f, 0.5f];
    private static readonly float[] Other = [-0.9f, 0.2f, 0.8f];
    private static readonly float[] Target = [1f, 0f];

    private static Network TwoLayer() =>
        new Sequential(inputs: 3).Dense<Tanh>(4).Dense<Sigmoid>(2).Build(seed: 7);

    private static float[] GradientsOf(Network net)
    {
        var all = new List<float>();

        foreach (ILayer layer in net.Layers)
            for (int p = 0; p < layer.ParameterCount; p++)
                all.Add(layer.GetParameterGradient(p));

        return [.. all];
    }

    /// <summary>
    /// The headline guarantee: inference between accumulating and applying gradients must be inert.
    /// Under the old design this test failed — <c>Predict</c> overwrote the activation cache, and
    /// the gradients silently became those of a different example.
    /// </summary>
    [Fact]
    public void Predicting_between_accumulate_and_apply_does_not_change_the_gradients()
    {
        Network clean = TwoLayer();
        clean.ZeroGradients();
        clean.AccumulateGradients(Input, Target);
        float[] expected = GradientsOf(clean);

        Network disturbed = TwoLayer();
        disturbed.ZeroGradients();
        disturbed.AccumulateGradients(Input, Target);
        disturbed.Predict(Other);            // the intruder
        disturbed.Predict(Other);
        float[] actual = GradientsOf(disturbed);

        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Same guarantee for <see cref="Network.Loss"/>, which is the one <see cref="GradientCheck"/>
    /// leans on: it evaluates the loss twice per parameter while analytic gradients are live.
    /// </summary>
    [Fact]
    public void Evaluating_the_loss_between_accumulate_and_apply_does_not_change_the_gradients()
    {
        Network clean = TwoLayer();
        clean.ZeroGradients();
        clean.AccumulateGradients(Input, Target);
        float[] expected = GradientsOf(clean);

        Network disturbed = TwoLayer();
        disturbed.ZeroGradients();
        disturbed.AccumulateGradients(Input, Target);
        disturbed.Loss(Other, Target);
        float[] actual = GradientsOf(disturbed);

        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// The mid-training version of the same thing, end to end: interleaving predictions with
    /// training must not change what the network learns.
    /// </summary>
    [Fact]
    public void Predicting_during_training_does_not_change_the_learned_weights()
    {
        float[] inputs = [0, 0, 0, 1, 1, 0, 1, 1];
        float[] targets = [0, 1, 1, 0];

        var quiet = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);
        quiet.Train(inputs, targets, epochs: 200, learningRate: 0.5f);

        var noisy = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);
        noisy.Train(inputs, targets, epochs: 200, learningRate: 0.5f,
            onEpoch: (_, _) => noisy.Predict(inputs.AsSpan(2, 2)));

        for (int l = 0; l < quiet.Layers.Count; l++)
            for (int p = 0; p < quiet.Layers[l].ParameterCount; p++)
                Assert.Equal(quiet.Layers[l].GetParameter(p), noisy.Layers[l].GetParameter(p));
    }

    /// <summary>Backprop without a training forward pass has nothing to differentiate.</summary>
    [Fact]
    public void Backward_without_a_forward_train_is_rejected()
    {
        var layer = new Dense<Tanh>(inputs: 3, units: 2);

        var ex = Assert.Throws<InvalidOperationException>(
            () => layer.Backward(new float[2], new float[3]));

        Assert.Contains("ForwardTrain", ex.Message);
    }

    /// <summary>An inference forward pass must not arm a backward pass.</summary>
    [Fact]
    public void Inference_forward_does_not_satisfy_backward()
    {
        var layer = new Dense<Tanh>(inputs: 3, units: 2);
        layer.Initialize(new Random(1));

        layer.Forward(Input, new float[2]);

        Assert.Throws<InvalidOperationException>(() => layer.Backward(new float[2], new float[3]));
    }

    /// <summary>Each backward pass consumes exactly one forward pass; a second is a bug.</summary>
    [Fact]
    public void Backward_cannot_be_run_twice_against_one_forward_train()
    {
        var layer = new Dense<Tanh>(inputs: 3, units: 2);
        layer.Initialize(new Random(1));

        layer.ForwardTrain(Input, new float[2]);
        layer.Backward(new float[2], new float[3]);

        Assert.Throws<InvalidOperationException>(() => layer.Backward(new float[2], new float[3]));
    }

    /// <summary>
    /// <see cref="Dense{T}.ForwardBatch"/> is inference-only. It must not leave a cache behind:
    /// a backward pass after it would otherwise differentiate whichever example came last, which
    /// is exactly the silent wrong-example bug this class exists to prevent.
    /// </summary>
    [Fact]
    public void ForwardBatch_does_not_arm_a_backward_pass()
    {
        var layer = new Dense<Tanh>(inputs: 2, units: 3);
        layer.Initialize(new Random(42));

        layer.ForwardBatch([0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f], new float[4 * 3], count: 4);

        Assert.Throws<InvalidOperationException>(() => layer.Backward(new float[3], new float[2]));
    }

    /// <summary>Forward and ForwardTrain must differ only in what they cache, never in arithmetic.</summary>
    [Fact]
    public void Forward_and_forward_train_compute_identical_activations()
    {
        var layer = new Dense<Tanh>(inputs: 3, units: 4);
        layer.Initialize(new Random(9));

        var inference = new float[4];
        var training = new float[4];

        layer.Forward(Input, inference);
        layer.ForwardTrain(Input, training);

        Assert.Equal(inference, training);
    }
}
