using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.SkillDefinition;

var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

string aoaiEndpoint = config["AZUREOPENAI_ENDPOINT"]!;
string aoaiApiKey = config["AZUREOPENAI_API_KEY"]!;
string aoaiModel = "gpt35turbo";

// Initialize the kernel
IKernel kernel = Kernel.Builder
    .WithAzureChatCompletionService(aoaiModel, aoaiEndpoint, aoaiApiKey)
    .Build();

// Register functions with the kernel
kernel.RegisterCustomFunction(SKFunction.FromNativeFunction(
    () => $"{DateTime.UtcNow:r}",
    "DateTime", "Now",
    "Gets the current date and time"));
ISKFunction qa = kernel.CreateSemanticFunction("""
    The current date and time is {{ datetime.now }}.
    {{ $input }}
    """, maxTokens: 2000);

// Q&A loop
while (true)
{
    Console.Write("Question: ");
    Console.WriteLine(await qa.InvokeAsync(Console.ReadLine()!, kernel.Skills));
    Console.WriteLine();
}

// Q&A loop
while (true)
{
    Console.Write("Question: ");
    Console.WriteLine(await kernel.InvokeSemanticFunctionAsync(Console.ReadLine()!, maxTokens: 2000));
    Console.WriteLine();
}