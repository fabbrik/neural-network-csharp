using NN.Mnist;
using Xunit;

namespace NN.Tests;

/// <summary>
/// The name-to-loss lookup that model files depend on.
///
/// <para>It is exercised indirectly whenever a model round-trips, but it deserves direct tests
/// because it is the hinge the file format turns on: every name here is written into saved models
/// and can never be changed without invalidating them.</para>
/// </summary>
public class LossRegistryTests
{
    [Fact]
    public void Every_built_in_loss_resolves_by_its_own_name()
    {
        foreach (ILoss loss in new ILoss[] { MeanSquaredError.Instance, SoftmaxCrossEntropy.Instance })
            Assert.Same(loss, Losses.ByNameOrThrow(loss.Name));
    }

    /// <summary>
    /// These strings live in every model file ever saved. Changing one silently turns previously
    /// saved models into unloadable files, so they are pinned here as the format constants they
    /// are — the same reason <see cref="ILayer.Descriptor"/> is pinned.
    /// </summary>
    [Fact]
    public void The_persisted_names_are_stable()
    {
        Assert.Equal("mse", MeanSquaredError.Instance.Name);
        Assert.Equal("softmax-cross-entropy", SoftmaxCrossEntropy.Instance.Name);
    }

    [Fact]
    public void The_default_is_mean_squared_error()
    {
        Assert.Same(MeanSquaredError.Instance, Losses.Default);
    }

    [Fact]
    public void An_unknown_name_is_rejected_and_lists_what_is_available()
    {
        var ex = Assert.Throws<InvalidDataException>(() => Losses.ByNameOrThrow("hinge"));

        Assert.Contains("hinge", ex.Message);
        Assert.Contains("mse", ex.Message);
        Assert.Contains("softmax-cross-entropy", ex.Message);
    }

    /// <summary>The instances are shared, so identity comparisons in the rest of the suite hold.</summary>
    [Fact]
    public void The_built_in_losses_are_singletons()
    {
        Assert.Same(MeanSquaredError.Instance, MeanSquaredError.Instance);
        Assert.Same(SoftmaxCrossEntropy.Instance, SoftmaxCrossEntropy.Instance);
    }
}

/// <summary>
/// What can be checked about the MNIST dataset loader without a network connection.
///
/// <para><b>Deliberately partial.</b> <see cref="MnistData.TryLoad"/> downloads ~11 MB and writes
/// a cache, so unit-testing it would mean either a network dependency in <c>dotnet test</c> or a
/// mock elaborate enough to test itself. The parsing it delegates to <see cref="Idx"/> is fully
/// covered in <c>IdxTests</c> against in-memory bytes, and the download-and-cache path is covered
/// by the CI smoke run of the demo. What remains testable here is the contract around the cache,
/// which is what callers actually branch on.</para>
/// </summary>
public class MnistDataTests
{
    [Fact]
    public void The_cache_lives_outside_the_working_tree()
    {
        string cache = MnistData.CacheDirectory;

        Assert.False(string.IsNullOrWhiteSpace(cache));
        Assert.True(Path.IsPathRooted(cache), $"expected an absolute path, got '{cache}'");

        // Downloaded data is not source. Caching it inside the repository would put 11 MB in
        // everyone's working tree and invite it into a commit.
        Assert.DoesNotContain("neural-network-csharp", cache);
        Assert.Contains("nn-mnist", cache);
    }

    /// <summary>
    /// <see cref="MnistData.IsCached"/> must agree with what is actually on disk — it is the flag
    /// a caller uses to decide whether a run needs the network at all.
    /// </summary>
    [Fact]
    public void Is_cached_reflects_whether_all_four_files_are_present()
    {
        string[] required =
        [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ];

        bool allPresent = required.All(f => File.Exists(Path.Combine(MnistData.CacheDirectory, f)));

        Assert.Equal(allPresent, MnistData.IsCached);
    }

    /// <summary>
    /// The split's counts must describe its own arrays. A mismatch here would send the demo
    /// indexing past the end of the pixel buffer.
    /// </summary>
    [Fact]
    public void A_split_reports_counts_consistent_with_its_arrays()
    {
        var split = new MnistData.Split(
            TrainX: new float[2 * Idx.PixelCount], TrainY: new float[2 * 10], TrainDigits: [1, 2],
            TestX: new float[3 * Idx.PixelCount], TestY: new float[3 * 10], TestDigits: [3, 4, 5]);

        Assert.Equal(2, split.TrainCount);
        Assert.Equal(3, split.TestCount);
        Assert.Equal(split.TrainCount * Idx.PixelCount, split.TrainX.Length);
        Assert.Equal(split.TestCount * 10, split.TestY.Length);
    }
}
