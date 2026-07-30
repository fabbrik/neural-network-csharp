using Xunit;

namespace NN.Tests;

/// <summary>
/// Mismatched inputs and targets are the mistake a reader assembling their own dataset by hand
/// will actually make. Caught at the call, they name the problem; uncaught, they surface as a
/// slice exception thousands of examples into training — or, when the lengths happen to divide
/// evenly, as silent training on misaligned rows.
/// </summary>
public class ValidationTests
{
    private static Network TwoInputsOneOutput() =>
        new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);

    [Fact]
    public void Training_on_inputs_and_targets_of_mismatched_length_is_rejected()
    {
        var net = TwoInputsOneOutput();

        // Four targets, but only three examples' worth of inputs.
        var ex = Assert.Throws<ArgumentException>(
            () => net.Train([0, 0, 0, 1, 1, 0], [0, 1, 1, 0], epochs: 1));

        Assert.Contains("4 examples", ex.Message);
        Assert.Contains("8", ex.Message);   // the expected input length
    }

    /// <summary>
    /// The nastier variant: the lengths are both valid multiples, so nothing would throw — the
    /// network would simply train on the wrong rows.
    /// </summary>
    [Fact]
    public void Training_on_a_silently_misaligned_dataset_is_rejected()
    {
        var net = TwoInputsOneOutput();

        Assert.Throws<ArgumentException>(
            () => net.Train([0, 0, 0, 1, 1, 0, 1, 1, 0, 0], [0, 1, 1, 0], epochs: 1));
    }

    [Fact]
    public void Training_targets_that_do_not_divide_into_outputs_are_rejected()
    {
        var net = new Sequential(inputs: 2).Dense<Sigmoid>(3).Build(seed: 1);

        var ex = Assert.Throws<ArgumentException>(
            () => net.Train([0, 0, 1, 1], [0, 1, 1, 0], epochs: 1));

        Assert.Contains("multiple", ex.Message);
    }

    [Fact]
    public void Training_with_no_samples_is_rejected()
    {
        Assert.Throws<ArgumentException>(() => TwoInputsOneOutput().Train([], [], epochs: 1));
    }

    [Fact]
    public void Training_for_a_non_positive_number_of_epochs_is_rejected()
    {
        var net = TwoInputsOneOutput();

        Assert.Throws<ArgumentOutOfRangeException>(
            () => net.Train([0, 0, 1, 1], [0, 1], epochs: 0));
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void Training_with_an_invalid_learning_rate_is_rejected(float learningRate)
    {
        var net = TwoInputsOneOutput();

        Assert.Throws<ArgumentOutOfRangeException>(
            () => net.Train([0, 0, 1, 1], [0, 1], epochs: 1, learningRate: learningRate));
    }

    [Fact]
    public void Training_with_a_negative_batch_size_is_rejected()
    {
        var net = TwoInputsOneOutput();

        Assert.Throws<ArgumentOutOfRangeException>(
            () => net.Train([0, 0, 1, 1], [0, 1], epochs: 1, batchSize: -1));
    }

    [Fact]
    public void Predicting_with_the_wrong_input_width_is_rejected()
    {
        var net = TwoInputsOneOutput();

        var ex = Assert.Throws<ArgumentException>(() => net.Predict([1f]));

        Assert.Contains("2 inputs", ex.Message);
    }

    [Fact]
    public void Loss_with_the_wrong_target_width_is_rejected()
    {
        var net = TwoInputsOneOutput();

        var ex = Assert.Throws<ArgumentException>(() => net.Loss([1f, 0f], [1f, 0f]));

        Assert.Contains("1 targets", ex.Message);
    }

    [Fact]
    public void Accumulating_gradients_with_the_wrong_target_width_is_rejected()
    {
        var net = TwoInputsOneOutput();

        var ex = Assert.Throws<ArgumentException>(() => net.AccumulateGradients([1f, 0f], [1f, 0f]));

        Assert.Contains("1 targets", ex.Message);
    }

    [Fact]
    public void Perceptron_training_on_mismatched_lengths_is_rejected()
    {
        var p = new Perceptron(inputs: 2);

        var ex = Assert.Throws<ArgumentException>(() => p.Train([0, 0, 0, 1], [0, 0, 0, 1]));

        Assert.Contains("4 examples", ex.Message);
    }

    [Fact]
    public void Perceptron_training_with_no_samples_is_rejected()
    {
        Assert.Throws<ArgumentException>(() => new Perceptron(inputs: 2).Train([], []));
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void Perceptron_training_with_an_invalid_learning_rate_is_rejected(float learningRate)
    {
        var p = new Perceptron(inputs: 2);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => p.Train([0, 0, 1, 1], [0, 1], learningRate: learningRate));
    }

    [Fact]
    public void Forward_batch_rejects_mismatched_lengths()
    {
        var layer = new Dense<Tanh>(inputs: 2, units: 3);

        Assert.Throws<ArgumentException>(() => layer.ForwardBatch([0f, 0f, 1f], new float[6], count: 2));
        Assert.Throws<ArgumentException>(() => layer.ForwardBatch([0f, 0f, 1f, 1f], new float[5], count: 2));
    }

    [Fact]
    public void Parameter_access_rejects_out_of_range_indices()
    {
        var layer = new Dense<Tanh>(inputs: 2, units: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() => layer.GetParameter(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.GetParameter(layer.ParameterCount));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.SetParameter(layer.ParameterCount, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.GetParameterGradient(layer.ParameterCount));
    }

    [Fact]
    public void Unit_weights_rejects_an_out_of_range_unit()
    {
        var layer = new Dense<Tanh>(inputs: 2, units: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() => layer.UnitWeights(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.UnitWeights(3));
    }
}
