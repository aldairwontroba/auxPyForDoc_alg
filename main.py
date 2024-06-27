import sys
import ctypes
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import tkinter as tk
import csv
import os
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model

#///////////////////////////////////////////////////////////////////////////////////////////
# Configurar o TensorFlow para usar a CPU
#keras.mixed_precision.set_global_policy("mixed_float16")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    visible_devices = tf.config.get_visible_devices('GPU')
    for device in visible_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Se você quiser garantir que a CPU seja usada, você pode definir o seguinte:
tf.config.set_visible_devices([], 'GPU')
#///////////////////////////////////////////////////////////////////////////////////////////

class MyStruct(ctypes.Structure):
    _fields_ = [
        ("tipo", ctypes.c_int),
        ("lenpos", ctypes.c_int),
        ("lensig", ctypes.c_int),
        ("h1h_M", ctypes.c_float),
        ("h1a_M", ctypes.c_float),
        ("h3h_M", ctypes.c_float),
        ("h3a_M", ctypes.c_float),
        ("pyh1h", ctypes.c_float * 4096),
        ("pyh3h", ctypes.c_float * 4096),
        ("pyh3a", ctypes.c_float * 4096),
        ("signal", ctypes.c_float * (40 * 4096)),
    ]

    def copy(self):
        # Criar uma nova instância de MyStruct
        new_struct = MyStruct()

        # Copiar os valores dos campos da estrutura original para a nova estrutura
        new_struct.tipo = self.tipo
        new_struct.lenpos = self.lenpos
        new_struct.lensig = self.lensig
        new_struct.h1h_M = self.h1h_M
        new_struct.h1a_M = self.h1a_M
        new_struct.h3h_M = self.h3h_M
        new_struct.h3a_M = self.h3a_M

        # Copiar os valores dos arrays
        new_struct.pyh1h[:] = self.pyh1h[:]
        new_struct.pyh3h[:] = self.pyh3h[:]
        new_struct.pyh3a[:] = self.pyh3a[:]
        new_struct.signal[:] = self.signal[:]

        return new_struct

#///////////////////////////////////////////////////////////////////////////////////////////
def creatComProcess():
    # Definir o tipo de dados necessários
    LPVOID = ctypes.c_void_p
    SIZE_T = ctypes.c_size_t
    HANDLE = LPVOID  # Um HANDLE geralmente é um ponteiro void
    LPCWSTR = ctypes.c_wchar_p
    DWORD = ctypes.c_ulong

    # Cargar a biblioteca Kernel32.dll
    kernel32 = ctypes.WinDLL('kernel32.dll')

    # Definir o protótipo de função necessario para OpenFileMappingW
    OpenFileMappingW = kernel32.OpenFileMappingW
    OpenFileMappingW.restype = HANDLE
    OpenFileMappingW.argtypes = [DWORD, ctypes.c_bool, LPCWSTR]

    # Parâmetros para OpenFileMappingW
    dwDesiredAccess = 0x000F001F
    bInheritHandle = False
    lpName = LPCWSTR("Value_Mapping_py")  # Substitua pelo nome real do objeto de mapeamento

    # Chamar OpenFileMappingW para abrir o objeto de mapeamento
    hFileMapping = OpenFileMappingW(dwDesiredAccess, bInheritHandle, lpName)

    if not hFileMapping:
        print("Erro ao abrir o objeto de mapeamento de arquivos compartilhados:", ctypes.GetLastError())
        return -1
    else:
        # Definir o protótipo de função necessario para MapViewOfFile
        MapViewOfFile = kernel32.MapViewOfFile
        MapViewOfFile.restype = LPVOID
        MapViewOfFile.argtypes = [HANDLE, DWORD, DWORD, DWORD, SIZE_T]

        # Parâmetros para MapViewOfFile
        dwFileOffsetHigh = 0
        dwFileOffsetLow = 0
        dwNumberOfBytesToMap = 0  # Mapear toda a seção

        # Chamar MapViewOfFile para mapear o objeto de mapeamento em memória
        lpBaseAddress = MapViewOfFile(hFileMapping, dwDesiredAccess, dwFileOffsetHigh, dwFileOffsetLow,
                                      dwNumberOfBytesToMap)

        if not lpBaseAddress:
            print("Erro ao mapear o objeto de mapeamento em memória:", ctypes.GetLastError())
            return -1
        else:
            # Leia a struct da memória mapeada
            struct = MyStruct.from_address(lpBaseAddress)

            # Não se esqueça de liberar a memória mapeada quando terminar
            # kernel32.UnmapViewOfFile(lpBaseAddress)

            # Nome do evento criado em C++
            nome_W_evento = "WorkEvent_py"
            nome_B_evento = "BackEvent_py"

            # Abra o evento
            w_evento_handle = ctypes.windll.kernel32.OpenEventW(
                ctypes.c_uint(0x1F0003),  # Acesso total ao evento
                ctypes.c_bool(False),  # Não é necessário um handle herdado
                ctypes.c_wchar_p(nome_W_evento)  # Nome do evento
            )
            b_evento_handle = ctypes.windll.kernel32.OpenEventW(
                ctypes.c_uint(0x1F0003),  # Acesso total ao evento
                ctypes.c_bool(False),  # Não é necessário um handle herdado
                ctypes.c_wchar_p(nome_B_evento)  # Nome do evento
            )

            if not (w_evento_handle or b_evento_handle):
                print("Erro ao abrir o evento")
                return -1
            else:
                print("Conexao bem sucedida")
                my_events = (w_evento_handle, b_evento_handle, struct.copy())
                return my_events


def detect_abrupt_variation(signal):
    abrupt_changes = []
    for i in range(1, len(signal)):
        change = signal[i] - signal[i-1]
        if change < -0.3:
            abrupt_changes.append(i-1)
    if abrupt_changes:
        return abrupt_changes
    else:
        abrupt_changes.append(len(signal) - 1)
        return abrupt_changes


def selecionar_opcao(opcao):
    if opcao == "Descartar":
        print("Opção 'Descartar' selecionada. Fechando o programa.")
        fechar_janela()  # Função para fechar o programa
        return
    # Diretório onde os arquivos CSV serão salvos
    diretorio1 = 'E:/dataTesteDoc/train_result/'
    diretorio2 = 'E:/dataTesteDoc/train_sinal/'

    # Verificar quantos arquivos existem com o mesmo prefixo de opcao
    arquivos_existente = [nome_arquivo for nome_arquivo in os.listdir(diretorio1) if nome_arquivo.startswith(opcao)]

    # Determinar o número do próximo arquivo com base na quantidade de arquivos existentes
    numero_arquivo = len(arquivos_existente) + 1

    # Nome do arquivo
    nome_arquivo1 = f"{opcao}_{numero_arquivo}.csv"
    nome_arquivo2 = f"{opcao}_{numero_arquivo}.csv"

    # Caminho completo para o arquivo
    path1 = os.path.join(diretorio1, nome_arquivo1)
    path2 = os.path.join(diretorio2, nome_arquivo2)

    if(opcao == "NaoFAI"):
        res = 0
    elif(opcao == "Capacitor"):
        res = 1
    elif(opcao == "Inrush"):
        res = 2
    elif(opcao == "FAI-Alerta"):
        res = 3
    elif(opcao == "FAI-Deteccao"):
        res = 4
    else:
        res = 0

    # Salvar os dados no arquivo CSV
    write_to_csv(my_struct, path1, path2, res)

    print(f"Dados salvos com sucesso no arquivo '{nome_arquivo1}'.")

    fechar_janela()  # Função para fechar o programa
    return

def write_to_csv(struct, path1, path2, res):
    with open(path1, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["h1h", "h3h", "h3a", "tipo", "result", "p1", "p3", "h1M", "a1M", "h3M", "a3M"])
        writer.writerow([struct.pyh1h[0], struct.pyh3h[0], struct.pyh3a[0], struct.tipo, res, struct.lenpos,
                         struct.lensig, struct.h1h_M, struct.h1a_M, struct.h3h_M, struct.h3a_M])

        for i in range(1, 4096):
            writer.writerow([struct.pyh1h[i], struct.pyh3h[i], struct.pyh3a[i]])

    with open(path2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(40*4096):
            writer.writerow([struct.signal[i]])

def fechar_janela():
    janela.quit()
    janela.destroy()


def plotar_grafico(data1h, data3h, data3a, sinal, p1, p3, predictions):
    # Criar a janela principal
    global janela
    janela = tk.Tk()
    janela.title("Gráfico e Botões")

    # Criar e exibir o gráfico dentro da janela
    figura = plt.figure(figsize=(12, 8))

    # Subplot ocupando duas colunas na primeira linha
    ax1 = figura.add_subplot(4, 1, 1)
    ax1.plot(data1h, label='1h')
    ax1.axvline(x=p1, color='green', linestyle='--')
    ax1.axvline(x=p3, color='red', linestyle='--')
    ax1.legend()

    # Demais subplots
    ax2 = figura.add_subplot(4, 1, 2)
    ax2.plot(data3a, label='3a')
    ax2.axvline(x=p1, color='green', linestyle='--')
    ax2.axvline(x=p3, color='red', linestyle='--')
    ax2.legend()

    ax3 = figura.add_subplot(4, 1, 3)
    ax3.plot(data3h, label='3h')
    ax3.axvline(x=p1, color='green', linestyle='--')
    ax3.axvline(x=p3, color='red', linestyle='--')
    ax3.legend()

    # Plotar gráfico de barras com as previsões
    ax4 = figura.add_subplot(4, 1, 4)
    labels = ['Não FAI', 'FAI Alerta', 'FAI Detecção']
    values = predictions[0]
    ax4.bar(labels, values, color=['blue', 'orange', 'red'])
    ax4.set_title('Resultado da Previsão')
    ax4.set_xlabel('Categorias')
    ax4.set_ylabel('Probabilidade')
    for i, v in enumerate(values):
        ax4.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(figura, master=janela)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Adicionar a barra de ferramentas de navegação após a criação do canvas
    toolbar = NavigationToolbar2Tk(canvas, janela)
    toolbar.update()

    # Criar os botões
    nomes_botoes = ["Descartar", "NaoFAI", "Capacitor", "Inrush", "FAI-Alerta", "FAI-Deteccao"]
    for nome in nomes_botoes:
        botao = tk.Button(janela, text=nome, command=lambda opc=nome: selecionar_opcao(opc))
        botao.pack(side=tk.LEFT)

    janela.protocol("WM_DELETE_WINDOW", fechar_janela)

    # Iniciar o loop de eventos da janela
    janela.mainloop()


print(f"Iniciando script Python...")
handles = creatComProcess()
if handles == -1:
    print(f"Problema com o creatComProcess")
    quit()

w_evento_handle, b_evento_handle, my_struct = handles

# Extrair dados da estrutura
tipo = my_struct.tipo
p1 = my_struct.lenpos
p3 = my_struct.lensig
h1h = np.array(my_struct.pyh1h)  # Converter para numpy array
h3h = np.array(my_struct.pyh3h)  # Converter para numpy array
h3a = np.array(my_struct.pyh3a)  # Converter para numpy array
sinalIn = my_struct.signal[:]

h3a = np.abs(h3a - 180) / 180

# Substituir NaNs por 0.0
h1h = np.nan_to_num(h1h)
h3h = np.nan_to_num(h3h)
h3a = np.nan_to_num(h3a)

# Verificar se o denominador é zero e substituir por NaN
h1hz = np.where(np.isclose(h1h, 0), np.nan, h1h)
# Realizar a divisão
p3h = np.divide(h3h, h1hz)

# Aplicar condição para h1h menor que 0.1 e outras transformações
p3h[h1h < 0.05] = 0
p3h = np.where(p3h < 0.03, 0.0, p3h - 0.03)
h1h = np.where(h1h < 0.005, 0.0, h1h / 10)
h3h = np.where(h3h < 0.001, 0.0, h3h / 10)

# Pré-alocar arrays para p1g e p3g
p1g = np.zeros(4096)
p3g = np.zeros(4096)

# Utilizar operações vetorizadas para calcular p1g e p3g
k = np.arange(4096)
p1g = np.where(k < p1 - 1024, 0, np.where(k < p1, 1 - ((p1 - k) / 1024), 1))
p3g = np.where(k < p3 - 1024, 0, np.where(k < p3, 1 - ((p3 - k) / 1024), 1))

# Converter arrays resultantes para numpy arrays
h1hl = h1h
h3hl = h3h
p3hl = p3h
h3al = h3a
p1tl = p1g
p3tl = p3g

print(f"Carregar o modelo...")
# Carregar o modelo
model_path = 'C:/Users/Aldair/GoogleDrive/Doutorado/PROJETOS/pyProjects/auxPyForDoc_alg/meu_modelo.h5'
model = load_model(model_path)
print(f"Modelo carregado")

# Função para preparar dados para previsão
def preparar_dados_para_previsao(h1hl, h3hl, p3hl, h3al, p1tl, p3tl, tipo):
    # Adiciona uma dimensão extra para o batch size
    h1hl = np.expand_dims(h1hl, axis=0)
    h3hl = np.expand_dims(h3hl, axis=0)
    p3hl = np.expand_dims(p3hl, axis=0)
    h3al = np.expand_dims(h3al, axis=0)
    p1tl = np.expand_dims(p1tl, axis=0)
    p3tl = np.expand_dims(p3tl, axis=0)

    # Combina as features na última dimensão
    dados = np.stack((h1hl, h3hl, p3hl, h3al, p1tl, p3tl), axis=-1)
    dados = np.nan_to_num(dados)

    # Adiciona uma dimensão extra para o batch size na variável numérica
    tipo = np.array(tipo).reshape(-1, 1)
    tipo = np.expand_dims(tipo, axis=0)

    dados_completos = {'sequence': dados, 'numeric': tipo}
    return dados_completos

dados_completos = preparar_dados_para_previsao(h1hl, h3hl, p3hl, h3al, p1tl, p3tl, tipo)

# Realizar previsões
predictions = model.predict([dados_completos['sequence'], dados_completos['numeric']])

# Imprimir previsões
print(predictions)

plotar_grafico(my_struct.pyh1h, my_struct.pyh3h, my_struct.pyh3a, my_struct.signal, p1, p3, predictions)