using System.Net;
using System.Text.RegularExpressions;
using System.Text;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Text;
using Microsoft.SemanticKernel.Connectors.Memory.Sqlite;

var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

string aoaiEndpoint = config["AZUREOPENAI_ENDPOINT"]!;
string aoaiApiKey = config["AZUREOPENAI_API_KEY"]!;
string aoaiModel = "gpt35turbo";

// Initialize the kernel
IKernel kernel = Kernel.Builder
    .WithLoggerFactory(LoggerFactory.Create(builder => builder.AddConsole()))
    .WithAzureChatCompletionService(aoaiModel, aoaiEndpoint, aoaiApiKey)
    .WithAzureTextEmbeddingGenerationService("textembeddingada002", aoaiEndpoint, aoaiApiKey)
    .WithMemoryStorage(await SqliteMemoryStore.ConnectAsync("mydata.db"))
    .Build();

// Ensure we have embeddings for our document
ISemanticTextMemory memory = kernel.Memory;
IList<string> collections = await memory.GetCollectionsAsync();
string collectionName = "net7perf";
if (collections.Contains(collectionName))
{
    Console.WriteLine("Found database");
}
else
{
    using HttpClient client = new();
    string s = await client.GetStringAsync("https://devblogs.microsoft.com/dotnet/performance_improvements_in_net_7");
    List<string> paragraphs =
        TextChunker.SplitPlainTextParagraphs(
            TextChunker.SplitPlainTextLines(
                WebUtility.HtmlDecode(Regex.Replace(s, @"<[^>]+>|&nbsp;", "")),
                128),
            1024);
    for (int i = 0; i < paragraphs.Count; i++)
        await memory.SaveInformationAsync(collectionName, paragraphs[i], $"paragraph{i}");
    Console.WriteLine("Generated database");
}

// Create a new chat
IChatCompletion ai = kernel.GetService<IChatCompletion>();
ChatHistory chat = ai.CreateNewChat("You are an AI assistant that helps people find information.");
StringBuilder builder = new();

// Q&A loop
while (true)
{
    Console.Write("Question: ");
    string question = Console.ReadLine()!;

    builder.Clear();
    await foreach (var result in memory.SearchAsync(collectionName, question, limit: 3))
        builder.AppendLine(result.Metadata.Text);
    int contextToRemove = -1;
    if (builder.Length != 0)
    {
        builder.Insert(0, "Here's some additional information: ");
        contextToRemove = chat.Count;
        chat.AddUserMessage(builder.ToString());
    }

    chat.AddUserMessage(question);

    builder.Clear();
    await foreach (string message in ai.GenerateMessageStreamAsync(chat))
    {
        Console.Write(message);
        builder.Append(message);
    }
    Console.WriteLine();
    chat.AddAssistantMessage(builder.ToString());

    if (contextToRemove >= 0) chat.RemoveAt(contextToRemove);
    Console.WriteLine();
}