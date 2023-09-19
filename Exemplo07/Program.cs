using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Planning;
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
(string personName) => personName switch { "Jane" => 8, _ => 0 },
    "Demographics", "GetAge", "Gets the age of the person whose name is provided"));

var planner = new StepwisePlanner(kernel);
Plan p = planner.CreatePlan("Jane's dog is half her age. How old is the dog?");
Console.WriteLine($"Result: {await kernel.RunAsync(p)}");