using System.IO.Compression;
using NN.Mnist;
using Xunit;

namespace NN.Tests;

/// <summary>
/// Minimal PNG writer, so the decoder can be tested against files built in memory rather than
/// binary fixtures checked into the repository. Writing the encoder is also the cheapest way to
/// be sure the test is exercising real PNG structure — signature, chunks, CRCs, zlib, and the
/// per-scanline filter byte — rather than a convenient approximation of it.
/// </summary>
internal static class PngWriter
{
    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];

        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }

        return table;
    }

    private static uint Crc(byte[] bytes)
    {
        uint c = 0xFFFFFFFFu;
        foreach (byte b in bytes) c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    private static byte[] BigEndian(int v) => [(byte)(v >> 24), (byte)(v >> 16), (byte)(v >> 8), (byte)v];

    private static void Chunk(Stream s, string type, byte[] data)
    {
        s.Write(BigEndian(data.Length));

        var typed = new byte[4 + data.Length];
        System.Text.Encoding.ASCII.GetBytes(type).CopyTo(typed, 0);
        data.CopyTo(typed, 4);

        s.Write(typed);
        s.Write(BigEndian((int)Crc(typed)));
    }

    /// <param name="colorType">0 greyscale, 2 RGB, 6 RGBA.</param>
    /// <param name="samples">Row-major, <c>channels</c> bytes per pixel.</param>
    /// <param name="filter">Which PNG scanline filter to write; all five must decode.</param>
    public static byte[] Write(int width, int height, int colorType, byte[] samples, byte filter = 0)
    {
        int channels = colorType switch { 0 => 1, 2 => 3, 6 => 4, _ => throw new ArgumentException(null, nameof(colorType)) };
        int stride = width * channels;

        using var file = new MemoryStream();
        file.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });

        var ihdr = new List<byte>();
        ihdr.AddRange(BigEndian(width));
        ihdr.AddRange(BigEndian(height));
        ihdr.AddRange(new byte[] { 8, (byte)colorType, 0, 0, 0 });
        Chunk(file, "IHDR", [.. ihdr]);

        // Apply the requested filter, which the decoder must invert.
        var raw = new List<byte>();

        for (int y = 0; y < height; y++)
        {
            raw.Add(filter);

            for (int i = 0; i < stride; i++)
            {
                int value = samples[y * stride + i];
                int left = i >= channels ? samples[y * stride + i - channels] : 0;
                int up = y > 0 ? samples[(y - 1) * stride + i] : 0;
                int upLeft = i >= channels && y > 0 ? samples[(y - 1) * stride + i - channels] : 0;

                int encoded = filter switch
                {
                    0 => value,
                    1 => value - left,
                    2 => value - up,
                    3 => value - ((left + up) >> 1),
                    4 => value - Paeth(left, up, upLeft),
                    _ => throw new ArgumentOutOfRangeException(nameof(filter)),
                };

                raw.Add((byte)encoded);
            }
        }

        using var compressed = new MemoryStream();
        using (var z = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
            z.Write(raw.ToArray());

        Chunk(file, "IDAT", compressed.ToArray());
        Chunk(file, "IEND", []);

        return file.ToArray();
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}

public class ImageFileTests : IDisposable
{
    private readonly List<string> _temporary = [];

    private string TempFile(string extension, byte[] bytes)
    {
        string path = Path.Combine(Path.GetTempPath(), $"nn-img-{Guid.NewGuid():N}{extension}");
        File.WriteAllBytes(path, bytes);
        _temporary.Add(path);
        return path;
    }

    private string TempText(string extension, string content) =>
        TempFile(extension, System.Text.Encoding.ASCII.GetBytes(content));

    public void Dispose()
    {
        foreach (string path in _temporary)
            if (File.Exists(path)) File.Delete(path);
    }

    [Fact]
    public void Reads_an_eight_bit_greyscale_png()
    {
        byte[] samples = [0, 128, 255, 64];
        string path = TempFile(".png", PngWriter.Write(2, 2, colorType: 0, samples));

        GreyImage image = ImageFile.Load(path);

        Assert.Equal(2, image.Width);
        Assert.Equal(2, image.Height);
        Assert.Equal(0f, image[0, 0], 3);
        Assert.Equal(128 / 255f, image[1, 0], 3);
        Assert.Equal(1f, image[0, 1], 3);
    }

    /// <summary>
    /// Every PNG filter must invert correctly. This is the part of the decoder most likely to be
    /// subtly wrong, and a mistake in Paeth or Average produces a plausible but corrupted image
    /// rather than an error — which for a digit recognizer means a confident wrong answer.
    /// </summary>
    [Theory]
    [InlineData(0)]   // None
    [InlineData(1)]   // Sub
    [InlineData(2)]   // Up
    [InlineData(3)]   // Average
    [InlineData(4)]   // Paeth
    public void Decodes_every_scanline_filter_identically(byte filter)
    {
        const int Size = 9;
        var samples = new byte[Size * Size];

        // A gradient with structure in both directions, so filters that predict from the left and
        // filters that predict from above are each genuinely exercised.
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
                samples[y * Size + x] = (byte)((x * 23 + y * 41) % 256);

        string path = TempFile(".png", PngWriter.Write(Size, Size, colorType: 0, samples, filter));

        GreyImage image = ImageFile.Load(path);

        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
                Assert.Equal(samples[y * Size + x] / 255f, image[x, y], 3);
    }

    [Fact]
    public void Converts_rgb_to_luma_rather_than_a_flat_average()
    {
        // Pure green. A flat average would give 1/3; Rec. 601 luma gives 0.587.
        byte[] samples = [0, 255, 0];
        string path = TempFile(".png", PngWriter.Write(1, 1, colorType: 2, samples));

        Assert.Equal(0.587f, ImageFile.Load(path)[0, 0], 3);
    }

    /// <summary>Transparent areas are paper, not ink — they must composite to white.</summary>
    [Fact]
    public void Composites_transparency_over_white()
    {
        byte[] samples = [0, 0, 0, 0, 0, 0, 0, 255];   // transparent black, then opaque black
        string path = TempFile(".png", PngWriter.Write(2, 1, colorType: 6, samples));

        GreyImage image = ImageFile.Load(path);

        Assert.Equal(1f, image[0, 0], 3);
        Assert.Equal(0f, image[1, 0], 3);
    }

    [Fact]
    public void Reads_a_binary_pgm()
    {
        byte[] header = System.Text.Encoding.ASCII.GetBytes("P5\n2 2\n255\n");
        byte[] bytes = [.. header, 0, 255, 128, 64];

        GreyImage image = ImageFile.Load(TempFile(".pgm", bytes));

        Assert.Equal(2, image.Width);
        Assert.Equal(0f, image[0, 0], 3);
        Assert.Equal(1f, image[1, 0], 3);
    }

    [Fact]
    public void Reads_an_ascii_pgm_with_comments_in_the_header()
    {
        GreyImage image = ImageFile.Load(TempText(".pgm", "P2\n# drawn by hand\n2 2\n# max value next\n255\n0 255\n128 64\n"));

        Assert.Equal(2, image.Width);
        Assert.Equal(0f, image[0, 0], 3);
        Assert.Equal(1f, image[1, 0], 3);
    }

    /// <summary>In PBM, a set bit means black — the reverse of every other format here.</summary>
    [Fact]
    public void Reads_a_pbm_where_one_means_black()
    {
        GreyImage image = ImageFile.Load(TempText(".pbm", "P1\n2 1\n1 0\n"));

        Assert.Equal(0f, image[0, 0], 3);
        Assert.Equal(1f, image[1, 0], 3);
    }

    [Fact]
    public void Rejects_a_jpeg_by_name_rather_than_failing_obscurely()
    {
        string path = TempFile(".jpg", [0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0]);

        var ex = Assert.Throws<NotSupportedException>(() => ImageFile.Load(path));

        Assert.Contains("JPEG", ex.Message);
    }

    [Fact]
    public void Rejects_a_file_that_is_not_an_image()
    {
        string path = TempFile(".png", "definitely not a png"u8.ToArray());

        Assert.Throws<NotSupportedException>(() => ImageFile.Load(path));
    }

    [Fact]
    public void Rejects_a_truncated_png()
    {
        byte[] png = PngWriter.Write(4, 4, colorType: 0, new byte[16]);

        Assert.Throws<InvalidDataException>(() => ImageFile.Load(TempFile(".png", png[..(png.Length / 2)])));
    }
}

/// <summary>
/// The preprocessing contract. These matter because getting them wrong does not throw — it
/// produces a valid-looking 28x28 image that the network confidently misreads.
/// </summary>
public class DigitPreprocessorTests
{
    /// <summary>A dark digit on a light background, the way a scan or a drawing arrives.</summary>
    private static GreyImage PaperDigit(int size, int left, int top, int width, int height)
    {
        var pixels = new float[size * size];
        Array.Fill(pixels, 1f);   // white paper

        for (int y = top; y < top + height; y++)
            for (int x = left; x < left + width; x++)
                pixels[y * size + x] = 0f;   // dark ink

        return new GreyImage(size, size, pixels);
    }

    [Fact]
    public void Produces_exactly_one_mnist_image()
    {
        float[] result = DigitPreprocessor.ToMnist(PaperDigit(100, 30, 30, 20, 40));

        Assert.Equal(Idx.PixelCount, result.Length);
        Assert.All(result, v => Assert.InRange(v, 0f, 1f));
    }

    /// <summary>
    /// MNIST is white-on-black. Dark ink on light paper must come out inverted, or the network
    /// sees a solid white frame with a hole in it.
    /// </summary>
    [Fact]
    public void Inverts_dark_ink_on_light_paper()
    {
        float[] result = DigitPreprocessor.ToMnist(PaperDigit(100, 40, 30, 20, 40));

        // The ink became bright, and the corners — which were paper — became black.
        Assert.True(result.Max() > 0.9f, "ink should end up near white");
        Assert.Equal(0f, result[0], 3);
        Assert.Equal(0f, result[Idx.PixelCount - 1], 3);
    }

    /// <summary>An image already in MNIST convention must not be inverted back.</summary>
    [Fact]
    public void Leaves_white_on_black_alone()
    {
        var pixels = new float[100 * 100];   // black background
        for (int y = 30; y < 70; y++)
            for (int x = 40; x < 60; x++)
                pixels[y * 100 + x] = 1f;    // white ink

        float[] result = DigitPreprocessor.ToMnist(new GreyImage(100, 100, pixels));

        Assert.True(result.Max() > 0.9f);
        Assert.Equal(0f, result[0], 3);
    }

    /// <summary>
    /// The digit must be scaled up to fill the frame regardless of how much empty margin the
    /// source had — otherwise a digit photographed from a distance becomes a few grey pixels.
    /// </summary>
    [Fact]
    public void Scales_a_small_digit_in_a_large_frame_up_to_fill_the_box()
    {
        // A 10-pixel mark in a 400-pixel frame: 2.5% of the width.
        float[] result = DigitPreprocessor.ToMnist(PaperDigit(400, 100, 100, 10, 10));

        int inked = result.Count(v => v > 0.5f);

        // Scaled to the 20x20 box it should cover far more than the ~1 pixel a naive resize gives.
        Assert.True(inked > 100, $"expected the digit to fill the box, only {inked} pixels are lit");
    }

    /// <summary>Aspect ratio is preserved: a tall thin 1 must not be stretched into a blob.</summary>
    [Fact]
    public void Preserves_aspect_ratio()
    {
        float[] result = DigitPreprocessor.ToMnist(PaperDigit(200, 90, 40, 8, 120));

        // Measure the lit bounding box in the result.
        int minX = 28, maxX = -1, minY = 28, maxY = -1;

        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                if (result[y * 28 + x] > 0.5f)
                {
                    minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
                    minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
                }

        int litWidth = maxX - minX + 1, litHeight = maxY - minY + 1;

        Assert.True(litHeight > litWidth * 3,
            $"a 8x120 stroke should stay tall and thin, got {litWidth}x{litHeight}");
    }

    /// <summary>
    /// Wherever the digit sat in the source, it ends up centred by centre of mass — which is what
    /// MNIST did, and what the network therefore expects.
    /// </summary>
    [Theory]
    [InlineData(5, 5)]
    [InlineData(150, 20)]
    [InlineData(60, 160)]
    public void Centres_the_digit_by_centre_of_mass(int left, int top)
    {
        float[] result = DigitPreprocessor.ToMnist(PaperDigit(200, left, top, 30, 30));

        double mass = 0, x = 0, y = 0;

        for (int row = 0; row < 28; row++)
            for (int column = 0; column < 28; column++)
            {
                float m = result[row * 28 + column];
                mass += m; x += m * column; y += m * row;
            }

        Assert.True(mass > 0, "the digit disappeared");

        // Within one pixel of centre, and no closer is achievable: the digit is placed at an
        // integer offset, so a centroid landing on a half-pixel cannot be improved on. MNIST
        // itself carries the same quantization.
        Assert.InRange(x / mass, 13.0, 15.0);
        Assert.InRange(y / mass, 13.0, 15.0);
    }

    /// <summary>A blank image is not an error — it has no ink, and produces no ink.</summary>
    [Fact]
    public void A_blank_image_produces_a_blank_result()
    {
        var white = new float[50 * 50];
        Array.Fill(white, 1f);

        float[] result = DigitPreprocessor.ToMnist(new GreyImage(50, 50, white));

        Assert.All(result, v => Assert.Equal(0f, v, 3));
    }

    /// <summary>The output must be directly consumable by a network trained on MNIST.</summary>
    [Fact]
    public void Output_feeds_straight_into_an_mnist_shaped_network()
    {
        var net = new Sequential(inputs: Idx.PixelCount).Dense<Tanh>(16).Dense<Sigmoid>(10).Build(seed: 1);

        float[] result = DigitPreprocessor.ToMnist(PaperDigit(120, 40, 30, 30, 50));

        Assert.Equal(net.Inputs, result.Length);
        Assert.Equal(10, net.Predict(result).Length);
    }
}
