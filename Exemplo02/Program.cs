using System.ComponentModel;

using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
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

OpenAIPromptExecutionSettings openAIPromptExecutionSettings = new()
{
    ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
};

// Register functions with the kernel
kernel.Plugins.AddFromType<Time>("datetime");

var qa = kernel.CreateFunctionFromPrompt("""
    Answer on the language of the request
    The current date and time is {{ datetime.now }}.
    {{ $input }}
    """, openAIPromptExecutionSettings);


// Q&A loop
while (true)
{
    Console.Write("Question: ");
    Console.WriteLine(await qa.InvokeAsync(kernel, new ()
    {
        { "Input", Console.ReadLine()! }
    }));
    Console.WriteLine();
}

class Time
{
    [KernelFunction("now")]
    [Description("Gets the current date and time")]
    public string Now() => $"{DateTime.UtcNow:r}";
}