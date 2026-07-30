namespace NN;

/// <summary>
/// Fluent builder for a sequential (strictly linear) stack of layers, in the style of Keras'
/// <c>Sequential</c> model.
///
/// <code>
/// var net = new Sequential(inputs: 2)
///     .Dense&lt;Tanh&gt;(4)
///     .Dense&lt;Sigmoid&gt;(1)
///     .Build(seed: 42);
/// </code>
///
/// The point of the builder is **input-size inference**: you state the network's input width
/// once, and each layer takes its input count from the previous layer's unit count. Constructing
/// <see cref="Network"/> directly requires stating both ends of every layer and keeping them
/// consistent by hand.
/// </summary>
public sealed class Sequential
{
    private readonly List<ILayer> _layers = [];
    private int _width;   // output width of the last layer added — the next layer's input count
    private ILoss? _loss;

    /// <param name="inputs">Number of features in one input example.</param>
    public Sequential(int inputs)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);

        _width = inputs;
    }

    /// <summary>Layers added so far.</summary>
    public IReadOnlyList<ILayer> Layers => _layers;

    /// <summary>The width the next added layer will take as its input count.</summary>
    public int CurrentWidth => _width;

    /// <summary>
    /// Appends a dense layer of <paramref name="units"/> units with activation
    /// <typeparamref name="TActivation"/>. Its input count is inferred from the previous layer.
    /// </summary>
    public Sequential Dense<TActivation>(int units) where TActivation : IActivation
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(units);

        _layers.Add(new Dense<TActivation>(_width, units));
        _width = units;
        return this;
    }

    /// <summary>
    /// Appends a linear output layer and scores the network with softmax cross-entropy — the
    /// standard setup for choosing among <paramref name="classes"/> mutually exclusive categories.
    ///
    /// <code>
    /// var net = new Sequential(inputs: 784)
    ///     .Dense&lt;Tanh&gt;(128)
    ///     .SoftmaxOutput(10)      // Dense&lt;Identity&gt;(10) + SoftmaxCrossEntropy
    ///     .Build(seed: 42);
    /// </code>
    ///
    /// <para>The layer is deliberately <c>Dense&lt;Identity&gt;</c>: softmax is applied by the
    /// loss, not the layer, because it needs every unit's value at once while an
    /// <see cref="IActivation"/> sees one at a time. Doing both in one call is what stops anyone
    /// from pairing cross-entropy with a squashed output layer, which would make its fused
    /// gradient quietly wrong. See <see cref="SoftmaxCrossEntropy"/>.</para>
    /// </summary>
    /// <param name="classes">Number of categories, and so the number of outputs.</param>
    public Sequential SoftmaxOutput(int classes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classes);

        Dense<Identity>(classes);
        _loss = SoftmaxCrossEntropy.Instance;

        return this;
    }

    /// <summary>
    /// Sets the loss explicitly. Only needed for combinations the shortcuts don't cover —
    /// <see cref="SoftmaxOutput"/> is the usual way to get a classifier, and mean squared error
    /// is the default otherwise.
    /// </summary>
    public Sequential WithLoss(ILoss loss)
    {
        ArgumentNullException.ThrowIfNull(loss);

        _loss = loss;
        return this;
    }

    /// <summary>
    /// Appends a pre-built layer — the escape hatch for layer types the fluent API doesn't cover.
    /// Its <see cref="ILayer.Inputs"/> must match the current width.
    /// </summary>
    public Sequential Add(ILayer layer)
    {
        ArgumentNullException.ThrowIfNull(layer);

        if (layer.Inputs != _width)
            throw new ArgumentException(
                $"Layer expects {layer.Inputs} inputs but the stack is {_width} wide here.", nameof(layer));

        _layers.Add(layer);
        _width = layer.Units;
        return this;
    }

    /// <summary>
    /// Builds the network: allocates buffers and randomizes weights.
    /// </summary>
    /// <param name="seed">Fixed by default so runs reproduce while you experiment.</param>
    public Network Build(int seed = 42)
    {
        if (_layers.Count == 0)
            throw new InvalidOperationException("Add at least one layer before building.");

        // No shape check here: every layer was either constructed at the current width or
        // validated against it by Add, so a mismatch cannot reach this point. The loss does get
        // checked against the output layer — see Network's constructor.
        return new Network(seed, _loss, [.. _layers]);
    }
}
