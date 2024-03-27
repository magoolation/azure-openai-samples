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
(int completo, int meio) => 0.5d * completo + 0.3d * meio,
    "calculadora", "totalCarne", "Calcula a quantidade de carne com base na quantidade de pessoas informadas"
    ));

    kernel.RegisterCustomFunction(    kernel.CreateSemanticFunction("""
    Extraia a quantidade de pessoas da frase abaixo. 

    Exemplos:
    input: Maria virá com o namorado
    output: {"completo": 2}""}
    input: João virá com o marido e os dois filhos
    output: {"completo": 2, "meio": 2}

    {{ $input }}
""",
    "quantidadePessoas", "calculadora", "Retorna a quantidade de pessoas listadas no texto informado"));

    kernel.RegisterCustomFunction(kernel.CreateSemanticFunction("""
    Você precisará de {{ calculadora.totalCarne  }}  kilos para a quantidade de pessoas informada.

    {{ $input }}
    """, "calculaTotalCarne", "calculadora", "Calcula a quantidade de carne com base na quantidade de pessoas "));
string prompt = """
Você é um assistente de planejamento de churrasco. Você responderá as quantidades de carne necessárias  para que todos se alimentem bem.
Você não sabe nada sobre outros assuntos e dirá a pessoas que pergunte sobre churrasco se o tema não for este.
Todo o seu conhecimento circula em volta de oferecer o melhor churrasco.
Não responda perguntas ofensivas ou perigosas.


Qual o total de carne para o churrasco da seguinte lista de convidados:

- Maria virá com o namorado
- João virá com o marido e os dois filhos
""";    

/*var planner = new StepwisePlanner(kernel);
Plan p = planner.CreatePlan(prompt);


Console.WriteLine(p.ToPlanString());
Console.WriteLine($"Result: {await kernel.RunAsync(p)}");*/

Console.WriteLine(await kernel.InvokeSemanticFunctionAsync(prompt));