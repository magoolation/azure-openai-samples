using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.Embeddings;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI.TextEmbedding;

var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

string aoaiEndpoint = config["AZUREOPENAI_ENDPOINT"]!;
string aoaiApiKey = config["AZUREOPENAI_API_KEY"]!;
var embeddingGen = new AzureTextEmbeddingGeneration("textembeddingada002", aoaiEndpoint, aoaiApiKey);

string input = "What is an amphibian?";
string[] examples =
{
    "What is an amphibian?",
    "Cos'è un anfibio?",
    "A frog is an amphibian.",
    "Frogs, toads, and salamanders are all examples.",
    "Amphibians are four-limbed and ectothermic vertebrates of the class Amphibia.",
    "They are four-limbed and ectothermic vertebrates.",
    "A frog is green.",
    "A tree is green.",
    "It's not easy bein' green.",
    "A dog is a mammal.",
    "A dog is a man's best friend.",
    "You ain't never had a friend like me.",
    "Rachel, Monica, Phoebe, Joey, Chandler, Ross",
};

// Generate embeddings for each piece of text
ReadOnlyMemory<float> inputEmbedding = await embeddingGen.GenerateEmbeddingAsync(input);
ReadOnlyMemory<float>[] exampleEmbeddings = await Task.WhenAll(examples.Select(example => embeddingGen.GenerateEmbeddingAsync(example)));

// Print the cosine similarity between the input and each example
float[] similarity = exampleEmbeddings.Select(e => CosineSimilarity(e.Span, inputEmbedding.Span)).ToArray();
similarity.AsSpan().Sort(examples.AsSpan(), (f1, f2) => f2.CompareTo(f1));
Console.WriteLine("Similarity Example");
for (int i = 0; i < similarity.Length; i++)
    Console.WriteLine($"{similarity[i]:F6}   {examples[i]}");

static float CosineSimilarity(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
{
    float dot = 0, xSumSquared = 0, ySumSquared = 0;

    for (int i = 0; i < x.Length; i++)
    {
        dot += x[i] * y[i];
        xSumSquared += x[i] * x[i];
        ySumSquared += y[i] * y[i];
    }

    return dot / (MathF.Sqrt(xSumSquared) * MathF.Sqrt(ySumSquared));
}