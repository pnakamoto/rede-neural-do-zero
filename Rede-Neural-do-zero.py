# IMPORTANDO BIBLIOTECAS
import numpy
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.ToTensor() # definindo a conversão de imagem para tensor

trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform) # Cria o dataset de treinamento
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # Cria um buffer de dados para treinamento

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform) # Cria o dataset de validação
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True) # Cria um buffer de dados para validação

dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)  # Usa next() para obter o próximo lote
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')  # Visualiza a primeira imagem no lote

print(imagens[0].shape)#para verificar as dimensões do tensor de cada imagem
print(etiquetas[0].shape)#para verificar as dimensões do tensor de cada etiqueta

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128) # camada de entrada, 784 neurônios que se ligam a 128
        self.linear2 = nn.Linear(128, 64) #c camada interna 1, 128 neurônios que se ligam a 64
        self.linear3 = nn.Linear(64, 10) # camada interna 2, 64 neurônios que se ligam a 10
        #para a camada de saida não é necessário definir nada pois só precisamos pegar o output da camada interna 2

    def forward(self,X):
        X = F.relu(self.linear1(X)) # função de ativação da camada de entrada para a camada interna 1
        X = F.relu(self.linear2(X)) # função de ativação da camada interna 1 para a camada interna 2
        X = self.linear3(X) # Função de ativação da camada interna 2 para a camada de saída, nesse caso f(x) = x
        return F.log_softmax(X, dim=1) # dados utilizados para calcular a perda

def treino(modelo, trainloader, device):

    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5) # define a política de atualização dos pesos e das bias
    inicio = time() # timer para sabermos quanto tempo levou o treino

    criteiro = nn.NLLLoss() # definindo o criterio para calcular a perda
    EPOCHS = 30 # numero de epochs que o algoritmo rodará
    modelo.train() # ativando o modo de treinamento do modelo

    for epoch in range(EPOCHS):
      perda_acumulada = 0 # inicialização da perda acumulada da epoch em questão

      for imagens, etiquetas in trainloader:

        imagens = imagens.view(imagens.shape[0], -1) # convertendo as imagens para "vetores" de 28*28 casas para ficaram compativeis com a
        otimizador.zero_grad() # zerando os gradientes por conta do ciclo anterior

        output = modelo(imagens.to(device)) # colocando os dados no modelo
        perda_instantanea.backwar() # back porpagation a partir da perda

        otimizador.step() # atualizando os pesos e a bias

        perda_acumulada += perda_instantanea.item() # atualização da perda acumulada

      else:
          print("Epoch {} - Perda Resultante: {}".format(epoch+1, perda_acumulada/len(trainloader)))
      print("n\Tempo de treino (em minutos) =", (time()-inicio)/60)
      
def validacao(modelo, valloader, device):
    conta_corretas, conta_todas = 0, 0

    with torch.no_grad():  # Desativa o autograd para economizar memória e acelerar
        for imagens, etiquetas in valloader:
            # Mova imagens e etiquetas para o dispositivo (CPU ou GPU)
            imagens, etiquetas = imagens.to(device), etiquetas.to(device)

            # Achatar as imagens de 28x28 para 784 (se necessário)
            imagens = imagens.view(imagens.shape[0], -1)

            # Output do modelo em escala logarítmica
            logps = modelo(imagens)

            # Converte as previsões para probabilidades
            ps = torch.exp(logps)
            probas, preds = ps.max(dim=1)  # Pega a maior probabilidade e a previsão correspondente

            # Compara as previsões com os valores corretos
            conta_corretas += (preds == etiquetas).sum().item()
            conta_todas += etiquetas.size(0)

    # Evita divisão por zero
    if conta_todas > 0:
        print("Total de imagens testadas =", conta_todas)
        print("\nPrecisão do modelo = {:.2f}%".format(conta_corretas * 100 / conta_todas))
    else:
        print("Nenhuma imagem foi testada!")

modelo = Modelo() # inicializa o modelo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # modelo rodará na GPU se possível
modelo.to(device)
