using BenchmarkDotNet.Running;

// Runs every benchmark class in this assembly. Filter from the command line, e.g.
//   dotnet run -c Release --project bench/NN.Bench -- --filter '*DotProduct*'
BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);

/// <summary>Anchor type for <see cref="BenchmarkSwitcher.FromAssembly"/>.</summary>
public partial class Program;
