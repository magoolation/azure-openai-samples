using Azure.AI.OpenAI;
using Azure;
using System.Net;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Text;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Memory;

using Microsoft.SemanticKernel.Text;
using Microsoft.SemanticKernel.Connectors.Memory.Qdrant;

var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

string aoaiEndpoint = config["AZUREOPENAI_ENDPOINT"]!;
string aoaiApiKey = config["AZUREOPENAI_API_KEY"]!;
string aoaiModel = "gpt35turbo16k";

// Initialize the kernel
var aoai = new OpenAIClient(new Uri(aoaiEndpoint), new AzureKeyCredential(aoaiApiKey));
IKernel kernel = Kernel.Builder
    .WithLoggerFactory(LoggerFactory.Create(builder => builder.AddConsole()))
    .WithAzureChatCompletionService(aoaiModel, aoai)
    .WithAzureTextEmbeddingGenerationService("textembeddingada002", aoaiEndpoint, aoaiApiKey)
    .WithMemoryStorage(new QdrantMemoryStore("http://localhost:6333/", 1536))
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
StringBuilder builder = new();
var chatCompletionsOptions = new ChatCompletionsOptions();
chatCompletionsOptions.Messages.Add(new ChatMessage(ChatRole.System, "You are an AI assistant that helps people find information."));
chatCompletionsOptions.Functions.Add(new FunctionDefinition("get_person_age")
{
    Description = "Gets the age of the named person",
    Parameters = BinaryData.FromString("""
                 {
                     "type":"object",
                     "properties":{
                         "name":{ 
                             "type":"string",
                             "description":"The name of a person"
                         }
                     },
                     "required": ["name"]
                 }
                 """)
});

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
        contextToRemove = chatCompletionsOptions.Messages.Count;
        chatCompletionsOptions.Messages.Add(new ChatMessage(ChatRole.User, builder.ToString()));
    }

    chatCompletionsOptions.Messages.Add(new ChatMessage(ChatRole.User, question));

    Response<ChatCompletions> response;
    ChatChoice c;
    while (true)
    {
        response = await aoai.GetChatCompletionsAsync(aoaiModel, chatCompletionsOptions);
        c = response.Value.Choices[0];
        if (c.FinishReason == CompletionsFinishReason.FunctionCall)
        {
            switch (c.Message.FunctionCall.Name)
            {
                case "get_person_age":
                    int age = JsonNode.Parse(c.Message.FunctionCall.Arguments)?["name"]?.ToString() switch
                    {
                        "Elsa" => 21,
                        "Anna" => 18,
                        _ => -1,
                    };
                    chatCompletionsOptions.Messages.Add(c.Message);
                    chatCompletionsOptions.Messages.Add(new ChatMessage(ChatRole.Function, $"{{ \"age\":{age} }}") { Name = "get_person_age" });
                    continue;
            }
        }
        break;
    }

    Console.WriteLine(c.Message.Content);
    chatCompletionsOptions.Messages.Add(c.Message);
    if (contextToRemove >= 0) chatCompletionsOptions.Messages.RemoveAt(contextToRemove);
}