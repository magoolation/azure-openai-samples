using System.Text;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;

var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

string aoaiEndpoint = config["AZUREOPENAI_ENDPOINT"]!;
string aoaiApiKey = config["AZUREOPENAI_API_KEY"]!;
string aoaiModel = "gpt35turbo";

// Initialize the kernel
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
Kernel kernel = Kernel.CreateBuilder()    
    .AddAzureOpenAIChatCompletion(aoaiModel, aoaiEndpoint, aoaiApiKey)
    .AddAzureOpenAITextEmbeddingGeneration("textembeddingada002", aoaiEndpoint, aoaiApiKey)
    //.AddMem
    .Build();
#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

// Register functions with the kernel
kernel.RegisterCustomFunction(SKFunction.FromNativeFunction(
(string orderNumber) => orderNumber switch { "123" => "Seu pedido está a caminho.", _ => string.Empty },
    "Pedidos", "orderStatus", "Pega o status de um determinado pedido"));

ISKFunction qa = kernel.CreateSemanticFunction("""
    O status do seu pedido {{ $input }} é  {{ Pedidos.orderStatus }}.    
    """, maxTokens: 2000);


// Create a new chat
IChatCompletion ai = kernel.GetService<IChatCompletion>();
ChatHistory chat = ai.CreateNewChat("""
     Você é um assistente virtual de uma loja online e auxilia os clientes com infromações sobre seus pedidos.
     Você não sabe nada além de responder sobre os pedidos dos clientes.
     Caso a pergunta seja sobre qualquer outro assuunto diga que adoraria ajudar mas que só consegue falar sobre os pedidos da loja
     """);

var planner = new StepwisePlanner(kernel);

StringBuilder builder = new();

// Q&A loop
while (true)
{
    Console.Write("Question: ");
    string question = Console.ReadLine()!;

    chat.AddUserMessage(question);
    var p = planner.CreatePlan(question);

    var message = await kernel.RunAsync(p);
    builder.Clear();
    builder.Append(message);
        
    Console.WriteLine();
    chat.AddAssistantMessage(builder.ToString());
}