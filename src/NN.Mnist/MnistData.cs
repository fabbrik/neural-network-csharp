using System.IO.Compression;

namespace NN.Mnist;

/// <summary>
/// Fetches the MNIST dataset, caching it on disk so the download happens at most once.
///
/// <para>The repository deliberately ships no data files: MNIST is ~11 MB, and a teaching repo
/// that stays cloneable in a second is worth more than one that always works offline. The
/// trade-off is handled rather than ignored — a machine with no network gets a clear explanation
/// and a skipped demo, never a stack trace.</para>
/// </summary>
public static class MnistData
{
    /// <summary>One train/test split, ready to hand to <see cref="Network.Train"/>.</summary>
    /// <param name="TrainX">Training pixels, row-major, 784 floats per image, scaled 0–1.</param>
    /// <param name="TrainY">Training targets, one-hot, 10 floats per image.</param>
    /// <param name="TrainDigits">Training labels as plain digits, for reporting.</param>
    /// <param name="TestX">Test pixels, same layout.</param>
    /// <param name="TestY">Test targets, same layout.</param>
    /// <param name="TestDigits">Test labels as plain digits.</param>
    public record Split(
        float[] TrainX, float[] TrainY, byte[] TrainDigits,
        float[] TestX, float[] TestY, byte[] TestDigits)
    {
        public int TrainCount => TrainDigits.Length;
        public int TestCount => TestDigits.Length;
    }

    private static readonly string[] Files =
    [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ];

    /// <summary>
    /// Where the files come from. LeCun's original site now refuses automated requests, so these
    /// are the two mirrors the major frameworks use; the second is tried only if the first fails.
    /// </summary>
    private static readonly string[] Mirrors =
    [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ];

    /// <summary>
    /// Cache directory: <c>&lt;LocalApplicationData&gt;/nn-mnist</c>. Outside the repository on
    /// purpose — downloaded data is not source, and should survive a clean checkout.
    /// </summary>
    public static string CacheDirectory { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "nn-mnist");

    /// <summary>True when every file is already cached, so a run needs no network at all.</summary>
    public static bool IsCached =>
        Files.All(f => File.Exists(Path.Combine(CacheDirectory, f)));

    /// <summary>
    /// Loads MNIST, downloading and caching it if necessary.
    /// </summary>
    /// <param name="trainLimit">Cap on training images; 0 reads all 60,000.</param>
    /// <param name="testLimit">Cap on test images; 0 reads all 10,000.</param>
    /// <param name="log">Receives progress messages.</param>
    /// <returns>
    /// The split, or <c>null</c> if the data is neither cached nor downloadable. Returning null
    /// rather than throwing is the point: an offline machine should see the rest of the demo's
    /// explanation and a clear reason, not an unhandled exception.
    /// </returns>
    public static Split? TryLoad(int trainLimit = 0, int testLimit = 0, Action<string>? log = null)
    {
        log ??= _ => { };

        if (!IsCached && !TryDownload(log))
            return null;

        var (trainX, _) = Idx.ReadImages(OpenCached(Files[0]), trainLimit);
        var (trainY, trainDigits) = Idx.ReadLabels(OpenCached(Files[1]), trainLimit);
        var (testX, _) = Idx.ReadImages(OpenCached(Files[2]), testLimit);
        var (testY, testDigits) = Idx.ReadLabels(OpenCached(Files[3]), testLimit);

        return new Split(trainX, trainY, trainDigits, testX, testY, testDigits);
    }

    /// <summary>Opens a cached .gz file as a decompressed stream.</summary>
    private static Stream OpenCached(string name) =>
        new GZipStream(File.OpenRead(Path.Combine(CacheDirectory, name)), CompressionMode.Decompress);

    /// <summary>
    /// Downloads whatever is missing. Files land at a temporary path and are moved into place only
    /// once complete, so an interrupted download can never leave a half-file that looks cached.
    /// </summary>
    private static bool TryDownload(Action<string> log)
    {
        log($"MNIST not cached. Downloading (~11 MB) to {CacheDirectory}");

        try
        {
            Directory.CreateDirectory(CacheDirectory);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            log($"  cannot create the cache directory: {e.Message}");
            return false;
        }

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };

        foreach (string file in Files)
        {
            string destination = Path.Combine(CacheDirectory, file);
            if (File.Exists(destination)) continue;

            if (!TryDownloadOne(http, file, destination, log))
                return false;
        }

        log("  cached — later runs need no network");
        return true;
    }

    private static bool TryDownloadOne(HttpClient http, string file, string destination, Action<string> log)
    {
        foreach (string mirror in Mirrors)
        {
            string temporary = destination + ".partial";

            try
            {
                byte[] bytes = http.GetByteArrayAsync(mirror + file).GetAwaiter().GetResult();

                File.WriteAllBytes(temporary, bytes);
                File.Move(temporary, destination, overwrite: true);

                log($"  {file}  {bytes.Length / 1024f / 1024f:F1} MB");
                return true;
            }
            catch (Exception e) when (e is HttpRequestException or TaskCanceledException or IOException)
            {
                log($"  {file} from {new Uri(mirror).Host} failed: {e.Message}");
                TryDelete(temporary);
            }
        }

        return false;
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
        }
        catch (IOException)
        {
            // A leftover .partial is harmless — it is never read, and the next run overwrites it.
        }
    }
}
