using Xunit;

namespace NN.Tests;

public class SequentialTests
{
    [Fact]
    public void Infers_each_layers_input_size_from_the_previous_layer()
    {
        var net = new Sequential(inputs: 3)
            .Dense<ReLU>(5)
            .Dense<Identity>(2)
            .Build();

        Assert.Equal(3, net.Layers[0].Inputs);
        Assert.Equal(5, net.Layers[0].Units);
        Assert.Equal(5, net.Layers[1].Inputs);
        Assert.Equal(2, net.Layers[1].Units);
        Assert.Equal(3 * 5 + 5 + 5 * 2 + 2, net.ParameterCount);
    }

    [Fact]
    public void Produces_the_same_network_as_the_direct_constructor()
    {
        var built = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);
        var direct = new Network(42, new Dense<Tanh>(2, 4), new Dense<Sigmoid>(4, 1));

        Assert.Equal(direct.ParameterCount, built.ParameterCount);

        for (int l = 0; l < direct.Layers.Count; l++)
            for (int p = 0; p < direct.Layers[l].ParameterCount; p++)
                Assert.Equal(direct.Layers[l].GetParameter(p), built.Layers[l].GetParameter(p));
    }

    [Fact]
    public void Building_with_no_layers_is_rejected()
    {
        Assert.Throws<InvalidOperationException>(() => new Sequential(inputs: 2).Build());
    }

    [Fact]
    public void Adding_a_layer_of_the_wrong_width_is_rejected()
    {
        var model = new Sequential(inputs: 2).Dense<Tanh>(4);

        var ex = Assert.Throws<ArgumentException>(() => model.Add(new Dense<ReLU>(7, 3)));
        Assert.Contains("7", ex.Message);
        Assert.Contains("4", ex.Message);
    }

    [Fact]
    public void Mismatched_layers_are_rejected_by_the_direct_constructor_too()
    {
        Assert.Throws<ArgumentException>(() =>
            new Network(42, new Dense<Tanh>(2, 4), new Dense<Sigmoid>(9, 1)));
    }

    [Fact]
    public void Zero_or_negative_sizes_are_rejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sequential(inputs: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sequential(2).Dense<Tanh>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense<Tanh>(-1, 4));
    }

    [Fact]
    public void Summary_lists_every_layer_and_the_total_parameter_count()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build();

        string summary = net.Summary();

        Assert.Contains("Dense<Tanh>", summary);
        Assert.Contains("Dense<Sigmoid>", summary);
        Assert.Contains("17", summary);   // 12 + 5
    }
}
