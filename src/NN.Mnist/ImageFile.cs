using System.IO.Compression;

namespace NN.Mnist;

/// <summary>
/// A greyscale image loaded from disk: <see cref="Pixels"/> is row-major, one float per pixel in
/// 0–1, with 0 black and 1 white.
/// </summary>
/// <param name="Width">Width in pixels.</param>
/// <param name="Height">Height in pixels.</param>
/// <param name="Pixels">Row-major intensities, <c>Width * Height</c> of them.</param>
public record GreyImage(int Width, int Height, float[] Pixels)
{
    /// <summary>Intensity at (x, y), 0 black to 1 white.</summary>
    public float this[int x, int y] => Pixels[y * Width + x];
}

/// <summary>
/// Reads PNG and Netpbm (PGM/PPM/PBM) images, with no dependency beyond the framework.
///
/// <para>PNG is here because it is what people actually have — export from any editor, or draw
/// something and save it. It is also more work than the rest of this file put together, and the
/// interesting part is that almost none of that work is decompression: .NET's
/// <see cref="ZLibStream"/> handles that. What's left is <i>unfiltering</i>, which is PNG's own
/// contribution and is explained at <see cref="Unfilter"/>.</para>
///
/// <para>Netpbm is here because it is trivial — a text header and raw bytes — and gives a format
/// you can produce from a shell script when you want to test something.</para>
///
/// <para>Not supported: interlaced PNG (rare, and a whole second layout scheme) and animation.
/// Both are reported clearly rather than mis-decoded.</para>
/// </summary>
public static class ImageFile
{
    private static ReadOnlySpan<byte> PngSignature => [137, 80, 78, 71, 13, 10, 26, 10];

    /// <summary>Loads an image, choosing the decoder by inspecting the file's first bytes.</summary>
    /// <exception cref="InvalidDataException">The file is corrupt, or in an unsupported variant.</exception>
    /// <exception cref="NotSupportedException">The format is not one this reader handles.</exception>
    public static GreyImage Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        byte[] bytes = File.ReadAllBytes(path);

        // Sniff the content rather than trusting the extension: a .png that is really a JPEG
        // should say so, not fail somewhere deep inside chunk parsing.
        if (bytes.Length >= 8 && bytes.AsSpan(0, 8).SequenceEqual(PngSignature))
            return DecodePng(bytes);

        if (bytes.Length >= 2 && bytes[0] == 'P' && bytes[1] is >= (byte)'1' and <= (byte)'6')
            return DecodeNetpbm(bytes);

        if (bytes.Length >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF)
            throw new NotSupportedException(
                $"'{Path.GetFileName(path)}' is a JPEG. This reader handles PNG and Netpbm " +
                "(.pgm/.ppm/.pbm) only — re-export as PNG.");

        throw new NotSupportedException(
            $"'{Path.GetFileName(path)}' is not a PNG or Netpbm image (unrecognized header).");
    }

    // ── PNG ──────────────────────────────────────────────────────────────────────────────────

    private static GreyImage DecodePng(byte[] bytes)
    {
        int width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
        byte[]? palette = null;
        var idat = new MemoryStream();
        bool seenHeader = false;

        int offset = PngSignature.Length;

        // A PNG is a sequence of length-prefixed, type-tagged chunks. Unknown chunks are skipped
        // by design — that is how the format stays extensible, and why gamma or text chunks from
        // whatever editor produced the file cause no trouble here.
        while (offset + 8 <= bytes.Length)
        {
            int length = ReadBigEndianInt32(bytes, offset);
            if (length < 0 || offset + 12 + length > bytes.Length)
                throw new InvalidDataException("PNG is truncated or declares an impossible chunk length.");

            string type = System.Text.Encoding.ASCII.GetString(bytes, offset + 4, 4);
            int data = offset + 8;

            switch (type)
            {
                case "IHDR":
                    width = ReadBigEndianInt32(bytes, data);
                    height = ReadBigEndianInt32(bytes, data + 4);
                    bitDepth = bytes[data + 8];
                    colorType = bytes[data + 9];
                    interlace = bytes[data + 12];
                    seenHeader = true;
                    break;

                case "PLTE":
                    palette = bytes[data..(data + length)];
                    break;

                case "IDAT":
                    // Image data may be split across any number of IDAT chunks; the compressed
                    // stream continues across the boundaries, so they must be concatenated first.
                    idat.Write(bytes, data, length);
                    break;

                case "IEND":
                    offset = bytes.Length;
                    continue;
            }

            offset += 12 + length;   // length + type + data + CRC
        }

        if (!seenHeader) throw new InvalidDataException("PNG has no IHDR chunk.");
        if (width <= 0 || height <= 0) throw new InvalidDataException($"PNG declares a {width}x{height} image.");
        if (idat.Length == 0) throw new InvalidDataException("PNG has no image data (no IDAT chunk).");

        if (interlace != 0)
            throw new NotSupportedException(
                "Interlaced (Adam7) PNG is not supported. Re-save without interlacing.");

        int channels = colorType switch
        {
            0 => 1,   // greyscale
            2 => 3,   // RGB
            3 => 1,   // palette index
            4 => 2,   // greyscale + alpha
            6 => 4,   // RGBA
            _ => throw new InvalidDataException($"PNG colour type {colorType} is not valid."),
        };

        if (colorType == 3 && palette is null)
            throw new InvalidDataException("PNG uses a palette but contains no PLTE chunk.");

        idat.Position = 0;
        using var inflater = new ZLibStream(idat, CompressionMode.Decompress);
        using var raw = new MemoryStream();
        inflater.CopyTo(raw);

        byte[] scanlines = Unfilter(raw.ToArray(), width, height, bitDepth, channels);

        return ToGrey(scanlines, width, height, bitDepth, colorType, channels, palette);
    }

    /// <summary>
    /// Reverses PNG's per-scanline filters.
    ///
    /// <para>This is the step that makes PNG compress well and the step everyone trips over.
    /// Before compression, each scanline is transformed by predicting every byte from its already
    /// known neighbours — the pixel to the left, the one above, or both — and storing only the
    /// difference. Smooth images turn into runs of near-zero bytes, which deflate flattens.</para>
    ///
    /// <para>The consequence is that decoding is inherently sequential: filter type 2 (Up) refers
    /// to the <i>reconstructed</i> scanline above, so line n cannot be recovered until line n−1
    /// is. Each scanline is prefixed with one byte naming which of the five filters it used.</para>
    /// </summary>
    private static byte[] Unfilter(byte[] raw, int width, int height, int bitDepth, int channels)
    {
        // Bytes per pixel, rounded up — sub-byte depths still predict from one byte back.
        int bpp = Math.Max(1, channels * bitDepth / 8);
        int stride = (width * channels * bitDepth + 7) / 8;

        long needed = (long)(stride + 1) * height;
        if (raw.Length < needed)
            throw new InvalidDataException(
                $"PNG image data is short: expected {needed} bytes after decompression, got {raw.Length}.");

        var output = new byte[stride * height];

        for (int y = 0; y < height; y++)
        {
            int filter = raw[y * (stride + 1)];
            int source = y * (stride + 1) + 1;
            int destination = y * stride;

            for (int i = 0; i < stride; i++)
            {
                int a = i >= bpp ? output[destination + i - bpp] : 0;              // left
                int b = y > 0 ? output[destination - stride + i] : 0;              // above
                int c = i >= bpp && y > 0 ? output[destination - stride + i - bpp] : 0;   // above-left

                int value = raw[source + i];

                output[destination + i] = (byte)(filter switch
                {
                    0 => value,                        // None
                    1 => value + a,                    // Sub
                    2 => value + b,                    // Up
                    3 => value + ((a + b) >> 1),       // Average
                    4 => value + Paeth(a, b, c),       // Paeth
                    _ => throw new InvalidDataException($"PNG scanline {y} uses unknown filter {filter}."),
                });
            }
        }

        return output;
    }

    /// <summary>
    /// PNG's Paeth predictor: of the pixel to the left, the one above, and the one above-left,
    /// pick whichever is closest to <c>a + b - c</c> — the value a locally linear surface would
    /// have. Cheap, and markedly better than the other filters on photographic content.
    /// </summary>
    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);

        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    /// <summary>Collapses whatever colour model the PNG used into a single intensity per pixel.</summary>
    private static GreyImage ToGrey(
        byte[] scanlines, int width, int height, int bitDepth, int colorType, int channels, byte[]? palette)
    {
        var pixels = new float[width * height];
        int stride = (width * channels * bitDepth + 7) / 8;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float r, g, b, alpha = 1f;

                if (colorType == 3)
                {
                    int index = ReadSample(scanlines, y * stride, x, bitDepth, 1, 0, scale: false);

                    if (palette is null || index * 3 + 2 >= palette.Length)
                        throw new InvalidDataException($"PNG palette index {index} is out of range.");

                    r = palette[index * 3] / 255f;
                    g = palette[index * 3 + 1] / 255f;
                    b = palette[index * 3 + 2] / 255f;
                }
                else
                {
                    float s0 = ReadSample(scanlines, y * stride, x, bitDepth, channels, 0, scale: true) / 255f;

                    if (channels >= 3)
                    {
                        r = s0;
                        g = ReadSample(scanlines, y * stride, x, bitDepth, channels, 1, scale: true) / 255f;
                        b = ReadSample(scanlines, y * stride, x, bitDepth, channels, 2, scale: true) / 255f;
                        if (channels == 4)
                            alpha = ReadSample(scanlines, y * stride, x, bitDepth, channels, 3, scale: true) / 255f;
                    }
                    else
                    {
                        r = g = b = s0;
                        if (channels == 2)
                            alpha = ReadSample(scanlines, y * stride, x, bitDepth, channels, 1, scale: true) / 255f;
                    }
                }

                // Rec. 601 luma: the eye is far more sensitive to green than to blue, and a flat
                // average would render a green digit lighter than it looks.
                float grey = 0.299f * r + 0.587f * g + 0.114f * b;

                // Composite over white. A digit drawn on a transparent canvas is conceptually ink
                // on paper, and treating "transparent" as black would make the whole background ink.
                pixels[y * width + x] = grey * alpha + (1f - alpha);
            }
        }

        return new GreyImage(width, height, pixels);
    }

    /// <summary>
    /// Reads one sample (channel <paramref name="channel"/> of pixel <paramref name="x"/>),
    /// handling the sub-byte bit depths PNG allows. With <paramref name="scale"/>, the result is
    /// normalized to 0–255 so callers need not care whether the source was 1, 2, 4, 8 or 16 bits.
    /// </summary>
    private static int ReadSample(
        byte[] data, int rowStart, int x, int bitDepth, int channels, int channel, bool scale)
    {
        if (bitDepth == 8)
            return data[rowStart + x * channels + channel];

        if (bitDepth == 16)
            // Take the high byte: MNIST-scale work has no use for 16 bits of precision.
            return data[rowStart + (x * channels + channel) * 2];

        // 1, 2 or 4 bits: several samples share a byte, most significant first.
        int index = x * channels + channel;
        int perByte = 8 / bitDepth;
        int b = data[rowStart + index / perByte];
        int shift = 8 - bitDepth * (index % perByte + 1);
        int value = (b >> shift) & ((1 << bitDepth) - 1);

        // Scale so full-on reads as 255 regardless of depth: at 1 bit, 1 means white, not 1/255.
        return scale ? value * 255 / ((1 << bitDepth) - 1) : value;
    }

    private static int ReadBigEndianInt32(byte[] b, int offset) =>
        (b[offset] << 24) | (b[offset + 1] << 16) | (b[offset + 2] << 8) | b[offset + 3];

    // ── Netpbm (PGM / PPM / PBM) ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes the Netpbm family: P1/P4 bitmap, P2/P5 greyscale, P3/P6 colour, in both the ASCII
    /// and binary variants. A header of a few whitespace-separated numbers, then the pixels.
    /// </summary>
    private static GreyImage DecodeNetpbm(byte[] bytes)
    {
        int format = bytes[1] - '0';
        int offset = 2;

        int width = ReadNetpbmInt(bytes, ref offset);
        int height = ReadNetpbmInt(bytes, ref offset);

        // P1 and P4 are bilevel and carry no maximum-value field.
        int max = format is 1 or 4 ? 1 : ReadNetpbmInt(bytes, ref offset);

        if (width <= 0 || height <= 0) throw new InvalidDataException($"Netpbm declares a {width}x{height} image.");
        if (max <= 0) throw new InvalidDataException($"Netpbm declares a maximum value of {max}.");

        var pixels = new float[width * height];
        bool binary = format >= 4;

        if (binary)
        {
            offset++;   // exactly one whitespace byte separates the header from binary data

            if (format == 4)
            {
                // P4 packs eight pixels per byte and pads each row to a byte boundary, so a row
                // is ceil(width / 8) bytes regardless of where the previous one ended. It also
                // uses 1 for *black* — the opposite of every other format here.
                int rowBytes = (width + 7) / 8;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int at = offset + y * rowBytes + x / 8;
                        if (at >= bytes.Length) throw new InvalidDataException("Netpbm image data is truncated.");

                        pixels[y * width + x] = ((bytes[at] >> (7 - x % 8)) & 1) == 1 ? 0f : 1f;
                    }
                }
            }
            else
            {
                int channels = format == 6 ? 3 : 1;

                for (int i = 0; i < pixels.Length; i++)
                {
                    int at = offset + i * channels;
                    if (at + channels > bytes.Length) throw new InvalidDataException("Netpbm image data is truncated.");

                    pixels[i] = channels == 3
                        ? (0.299f * bytes[at] + 0.587f * bytes[at + 1] + 0.114f * bytes[at + 2]) / max
                        : bytes[at] / (float)max;
                }
            }
        }
        else
        {
            int channels = format == 3 ? 3 : 1;

            for (int i = 0; i < pixels.Length; i++)
            {
                if (format == 1)
                {
                    pixels[i] = ReadNetpbmInt(bytes, ref offset) == 1 ? 0f : 1f;   // 1 is black
                    continue;
                }

                if (channels == 3)
                {
                    float r = ReadNetpbmInt(bytes, ref offset);
                    float g = ReadNetpbmInt(bytes, ref offset);
                    float b = ReadNetpbmInt(bytes, ref offset);
                    pixels[i] = (0.299f * r + 0.587f * g + 0.114f * b) / max;
                }
                else
                {
                    pixels[i] = ReadNetpbmInt(bytes, ref offset) / (float)max;
                }
            }
        }

        return new GreyImage(width, height, pixels);
    }

    /// <summary>
    /// Reads the next whitespace-separated integer, skipping <c>#</c> comments — which Netpbm
    /// permits anywhere in the header, including between the width and the height.
    /// </summary>
    private static int ReadNetpbmInt(byte[] bytes, ref int offset)
    {
        while (offset < bytes.Length)
        {
            if (bytes[offset] == '#')
                while (offset < bytes.Length && bytes[offset] != '\n') offset++;
            else if (char.IsWhiteSpace((char)bytes[offset]))
                offset++;
            else
                break;
        }

        int value = 0;
        bool any = false;

        while (offset < bytes.Length && bytes[offset] is >= (byte)'0' and <= (byte)'9')
        {
            value = value * 10 + (bytes[offset] - '0');
            offset++;
            any = true;
        }

        if (!any) throw new InvalidDataException("Netpbm header is malformed or the file is truncated.");

        return value;
    }
}
