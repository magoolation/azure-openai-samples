using Azure.AI.OpenAI;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;
using Microsoft.SemanticKernel;

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

// Q&A loop
while (true)
{
    Console.Write("Question: ");
    Console.WriteLine(await kernel.InvokePromptAsync(Console.ReadLine()!, new()
    {
        { "maxtokens", 2000}
    }));
    Console.WriteLine();
}