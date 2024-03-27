using System.Text;

using Azure.AI.OpenAI;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

string aoaiEndpoint = config["AZUREOPENAI_ENDPOINT"]!;
string aoaiApiKey = config["AZUREOPENAI_API_KEY"]!;
string aoaiModel = "gpt35turbo";

// Initialize the kernel
Kernel kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(aoaiModel, aoaiEndpoint, aoaiApiKey)
    .Build();

// Create a new chat
IChatCompletionService ai = kernel.GetRequiredService<IChatCompletionService>();
ChatHistory chat = new ChatHistory("You are an AI assistant that helps people find information.");

StringBuilder builder = new();

// Q&A loop
while (true)
{
    Console.Write("Question: ");
    chat.AddUserMessage(Console.ReadLine()!);

    builder.Clear();
    await foreach (var message in ai.GetStreamingChatMessageContentsAsync(chat))
    {
        Console.Write(message);
        builder.Append(message);
    }
    Console.WriteLine();
    chat.AddAssistantMessage(builder.ToString());

    Console.WriteLine();
}