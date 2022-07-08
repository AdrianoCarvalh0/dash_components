import os
import pickle
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def plot_vessel_map(vessel_map):
    """ Função que faz o plot do mapa do vaso. Mapeia os valores de zero sendo o mínimo e 60 como sendo o máximo

    Parâmetros:
    -----------
   vessel_map: object VesselMap
        instância do objeto VesselMap
    Retorno:
    -----------
        plota os valores das intensidades dos pixels do vaso sanguíneo.
    """
    plt.figure(figsize=[12, 10])
    plt.title("Map values Vmin=0 e Vmax=60")
    plt.xticks([])
    plt.yticks([])

    # o mapped_values, são os valores das intensidades dos pixels do vaso sanguíneo.
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)

    # mostra os valores do path1 mapeado em amarelo
    plt.plot(vessel_map.path1_mapped, c='yellow')

    # mostra os valores do path2 mapeado em amarelo
    plt.plot(vessel_map.path2_mapped, c='yellow')
    plt.show()



def plot_intensity_lines(vessel_map, half_size_vessel_map):
    """ Função que plota a intensidade da linha mediana, uma acima e uma abaixo dos valores mapeados

    Parâmetros:
    -----------
    vessel_map: object VesselMap
       instância do objeto VesselMap
    half_size_vessel_map: int
        metade inteira da divisão do tamanho dos valores mapeados por 2
    Retorno:
    -----------
        plote da intensidade da linha mediana, uma acima e uma abaixo dos valores mapeados
    """
    plt.figure(figsize=[12, 10])
    plt.title(
        f'Intensity of position in sections of the vessel {half_size_vessel_map - 1} , {half_size_vessel_map} and {half_size_vessel_map + 1}')

    # acima
    plt.plot(vessel_map.mapped_values[half_size_vessel_map - 1].flatten(),
             label=f'position:  {half_size_vessel_map - 1}')
    # linha do centro
    plt.plot(vessel_map.mapped_values[half_size_vessel_map].flatten(), label=f'position:  {half_size_vessel_map}')

    # linha abaixo
    plt.plot(vessel_map.mapped_values[half_size_vessel_map + 1].flatten(), label=f'position:  {half_size_vessel_map + 1}')

    plt.legend(loc='lower right')
    plt.xlabel('Positions')
    plt.ylabel('Intensity')
    plt.show()


def plot_fill_means_std_dev(means, std_dev):
    """ Função que plota a diferença entre a média e o desvio padrão

    Parâmetros:
    -----------
    means: ndarray, float
        média de todos os valores mapeados ao longo das linhas
    std_dev: ndarray, float
        desvio padrão de todos os valores mapeados ao longo das linhas
    Retorno:
    -----------
        plota a diferença entre a média e o desvio padrão
    """

    plt.figure(figsize=[12, 10])
    plt.title("Filling between the mean intensity and standard deviation")

    # mostra o sombreamento
    plt.fill_between(range(len(means)), means - std_dev, means + std_dev, alpha=0.3)

    # mostra a média
    plt.plot(range(len(means)), means)
    plt.show()


def plot_diameter_vessel(vessel_map):
    """ Função que plota o diâmetro dos vasos mapeados

    Parâmetros:
    -----------
    vessel_map: object VesselMap
       instância do objeto VesselMap
    Retorno:
    -----------
        plota o diâmetro dos vasos mapeados
    """
    plt.figure(figsize=[12, 10])

    # o diâmetro é o módulo da difrença entre os dois caminhos mapeados.
    diameter = np.abs(vessel_map.path1_mapped - vessel_map.path2_mapped)

    plt.title("Diameter of the vessel")
    plt.xlabel('Column index')
    plt.ylabel('Diameter')

    # o diâmetro é float, portanto necessitou do range(len)
    plt.plot(range(len(diameter)), diameter)
    plt.show()


def return_intensity_cols(vessel_map):
    """ Função que armazena todas as intensidades das colunas

    Parâmetros:
    -----------
    vessel_map: object VesselMap
       instância do objeto VesselMap
    Retorno:
    -----------
     intensity_cols_values_all: list, ndarray
        lista contendo todos os valores das intensidades das colunas em formato array numpy
    """

    # número de linhas e colunas do mapa do vaso
    num_rows, num_cols = vessel_map.mapped_values.shape

    intensity_cols_values_all = []

    # armazena todas as intensidades das colunas ao longo das linhas
    for i in range(num_cols):
        intensity_cols_values_all.append(vessel_map.mapped_values[0:num_rows, i])
    return intensity_cols_values_all


def return_clipping(vessel_map):
    """ Função que faz o recorte de uma imagem

    Parâmetros:
    -----------
    vessel_map: object VesselMap
       instância do objeto VesselMap
    Retorno:
    -----------
    clipping: ndarray, float
        imagem recortada mostrando a área em que o vaso se encontra. Nesta imagem é exibida apenas os valores mapeados
        com um padding de 1 pixel apenas.
    """
    padding = 1
    # linha mínima do path2
    line_min_path2 = int(np.min(vessel_map.path2_mapped) + padding)
    # linha máxima do path1
    line_max_path1 = int(np.max(vessel_map.path1_mapped) + padding)

    # todos os valores mapeados
    img_path = vessel_map.mapped_values

    # puxando o número de colunas da imagem
    _, num_cols = img_path.shape

    # o recorte é feito da linha minima e da linha máxima, e das colunas variando de 0 até o número de colunas existentes
    clipping = (img_path[line_min_path2:line_max_path1, 0:num_cols])
    return clipping



def plot_clipping(vessel_map):
    """ Função que plota uma imagem, com valores mínimos de zero e máximo de 60

    Parâmetros:
    -----------
    vessel_map: object VesselMap
       instância do objeto VesselMap
    Retorno:
    -----------
        plote da imagem recortada mostrando a área em que o vaso se encontra com um pixel de padding
    """
    # chama a função que retorna o recorte
    clipp = return_clipping(vessel_map)

    plt.figure(figsize=[8, 5])
    plt.title("Image clipping")
    plt.imshow(clipp, 'gray', vmin=0, vmax=60)
    plt.show()


def plot_intensity_cols_with_line_vessel(vessel_map):
    """ Função que plota a intensidade das colunas. Exibe também onde começa e termina o vaso
     através das barras centrais, perperdinculares ao eixo y

    Parâmetros:
    -----------
    vessel_map: object VesselMap
       instância do objeto VesselMap
    Retorno:
    -----------
        plota a intensidade das colunas e exibe as delimitações dos vasos à esquerda e à direita
    """
    array_min_path = []
    array_max_path = []

    # número de linhas e colunas dos valores mapeados
    num_rows, num_cols = vessel_map.mapped_values.shape

    # cálculo do diâmetro
    diameter = np.abs(vessel_map.path1_mapped - vessel_map.path2_mapped)

    # várias cores para alinhar a cor das colunas que serão exibidas com as v_lines que mostram a delimitação dos vasos
    colors = ['blue', 'green', 'red', 'orange', 'gray']

    # chama a função que pega todas as intensidades das colunas
    intensity_cols_values_all = return_intensity_cols(vessel_map)

    # Pegando a posição 0, 1/4, 1/2, 3/4, e final das colunas
    colunas_demarcadas = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]

    plt.figure(figsize=[12, 10])
    plt.title(
        f'Intensity of columns {colunas_demarcadas[0]} , {colunas_demarcadas[1]}  , {colunas_demarcadas[2]} , {colunas_demarcadas[3]} and {colunas_demarcadas[4]}')
    plt.xlabel('Line index')
    plt.ylabel('Intensity')
    for i in range(len(colunas_demarcadas)):
        # plota as posições existentes nas colunas demarcadas no vetor que contém todas as intensidades das colunas
        plt.plot(range(num_rows), intensity_cols_values_all[colunas_demarcadas[i]],
                 label=f'position:  {colunas_demarcadas[i]}', color=colors[i])
    plt.legend(loc='lower right')

    for j in range(len(colunas_demarcadas)):
        min_path = np.argmin(intensity_cols_values_all[colunas_demarcadas[j]])
        array_min_path.append(intensity_cols_values_all[colunas_demarcadas[j]][min_path])

        max_path = np.argmax(intensity_cols_values_all[colunas_demarcadas[j]])
        array_max_path.append(intensity_cols_values_all[colunas_demarcadas[j]][max_path])

        plt.vlines(vessel_map.path1_mapped[colunas_demarcadas[j]], array_min_path[j], array_max_path[j],
                   color=colors[j])
        plt.vlines(vessel_map.path2_mapped[colunas_demarcadas[j]], array_min_path[j], array_max_path[j],
                   color=colors[j])
    plt.show()


def plot_intensity_cols_with_line_vessel_normal(vessel_map, colunas_demarcadas=None):
    """ Função que plota a intensidade das colunas. Exibe também onde começa e termina o vaso
        através das barras centrais, perperdinculares ao eixo y. Aqui exibiremos algumas colunas específicas ao
        longo do vaso. As intensidades das colunas serão mantidas, mas o eixo será normalizado conforme a linha do
        centro.

    Parâmetros:
    -----------
    vessel_map: object VesselMap
        instância do objeto VesselMap
    colunas_demarcadas: NoneType
        o campo é None por padrão, sendo setado posteriormente para pegar 5 colunas ao longo do vaso. Se este
        parâmetro vier preenchido, as colunas serão as que forem passadas por parâmetro
    Retorno:
    -----------
        plota a intensidade das colunas e exibe as delimitações dos vasos à esquerda e à direita
    """
    num_rows, num_cols = vessel_map.mapped_values.shape

    if (colunas_demarcadas is None):
        # Mostrando a posição 0, 1/4, 1/2, 3/4, e final das colunas
        colunas_demarcadas = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]

    # recebe um vetor de cores
    colors = ['blue', 'green', 'red', 'orange', 'gray']

    # puxa todas as intensidades de todas as colunas
    intensity_cols_values_all = return_intensity_cols(vessel_map)

    # Resto inteiro do número de linhas dividido por 2
    linha_centro = num_rows // 2

    # vetor criado para armazenar as posições
    vet_num_rows = []
    for i in range(num_rows):
        # criando um vetor de tamanho de 27 posições
        vet_num_rows.append(i)

    l_chapeu = []
    for j in range(len(vet_num_rows)):
        # Neste for faço a adição no vetor criado anteriormente. Colocando as linhas divididas por 2, ==> lc = num_rows//2
        # Normalização pela linha do centro
        l_chapeu.append(vet_num_rows[j] - linha_centro)

    lfv_list = []
    liv_list = []
    diametro = []
    l2_chapeu_all = []
    for col in colunas_demarcadas:
        lfv = vessel_map.path2_mapped[col]
        liv = vessel_map.path1_mapped[col]
        lfv_list.append(lfv)
        liv_list.append(liv)
        # pega o último valor que foi adicionado na lista
        diametro.append(abs(lfv - liv))

        l2_chapeu = []
        for k in range(len(l_chapeu)):
            # Fórmula (L1'' = 2L'/(Lfv1-Liv1))
            l2_chapeu.append(2 * l_chapeu[k] / diametro[-1])
        l2_chapeu_all.append(l2_chapeu)

    plt.figure(figsize=[12, 10])
    for i in range(len(colunas_demarcadas)):
        # o parametro -o plota bolinhas
        plt.plot(l2_chapeu_all[i], intensity_cols_values_all[colunas_demarcadas[i]], '-o',
                 label=f'position:  {colunas_demarcadas[i]}', color=colors[i])
    plt.legend(loc='lower right')

    liv_list_vlines = []
    lfv_list_vlines = []
    # l = (vet_num_rows - linha_centro) /diametro
    for k in range(len(colunas_demarcadas)):
        formula1 = 2 * (liv_list[k] - linha_centro) / diametro[k]
        formula2 = 2 * (lfv_list[k] - linha_centro) / diametro[k]
        liv_list_vlines.append(formula1)
        lfv_list_vlines.append(formula2)

    array_min_path = []
    array_max_path = []

    for i in range(len(colunas_demarcadas)):
        min_path = np.argmin(intensity_cols_values_all[colunas_demarcadas[i]])
        array_min_path.append(intensity_cols_values_all[colunas_demarcadas[i]][min_path])

        max_path = np.argmax(intensity_cols_values_all[colunas_demarcadas[i]])
        array_max_path.append(intensity_cols_values_all[colunas_demarcadas[i]][max_path])
    plt.vlines(liv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors)
    plt.vlines(lfv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors)

    # VER
    plt.xlabel('Positions')
    plt.ylabel('Intensity')

    plt.legend(loc='lower right')
    plt.show()

def return_all_instisitys_normal(vessel_map):

    """ Função que retorna todas as intensidades normalizadas com a linha central

    Parâmetros:
    -----------
    vessel_map: object VesselMap
        instância do objeto VesselMap
    Retorno:
    -----------
    intensities_common_axis: ndarray, float
        vetor que contém as intensidades normalizadas
    l2_chapeu_axis: ndarray, float
        contém as informações sobre onde o eixo deve começar e terminar. Existe uma modificação na exibição do gráfic,
        ao invés de começar do ponto de origem [0,0]. Ele (ponto de origem) vai começar dependendo da quantidade de
        linhas que existirem. O menor valor encontrado
    """

    #EXPLICAR MELHOR ESTA FUNÇÃO

    num_rows, num_cols = vessel_map.mapped_values.shape

    # puxa todas as intensidades de todas as colunas
    intensity_cols_values_all = return_intensity_cols(vessel_map)

    # Resto inteiro do número de linhas dividido por 2
    linha_centro = num_rows // 2

    # vetor criado para armazenar as posições
    vet_num_rows = []
    for i in range(num_rows):
        # criando um vetor de tamanho de N posições
        vet_num_rows.append(i)

    l = []
    for j in range(len(vet_num_rows)):
        # Neste for faço a adição no vetor criado anteriormente. Colocando as linhas divididas por 2, ==> lc = num_rows//2
        l.append(vet_num_rows[j] - linha_centro)

    lfv_list = []
    liv_list = []
    diametro = []

    l_all = []
    for col in range(len(intensity_cols_values_all)):
        liv = vessel_map.path1_mapped[col]
        lfv = vessel_map.path2_mapped[col]
        liv_list.append(liv)
        lfv_list.append(lfv)
        # pega o último valor que foi adicionado na lista
        diametro.append(abs(lfv - liv))

        l2 = []
        for k in range(len(l)):
            # Fórmula (L1'' = 2L'/(Lfv1-Liv1))
            l2.append(2 * l[k] / diametro[-1])
        l_all.append(l2)

    l2_min, l2_max = np.min(l_all), np.max(l_all)

    l2_chapeu_axis = np.linspace(l2_min, l2_max, num_rows)

    # Create interpolating functions
    l2_chapeu_funcs = []
    for l2, intens in zip(l_all, intensity_cols_values_all):
        l2_chapeu_func = interp1d(l2, intens, kind='linear', bounds_error=False)
        l2_chapeu_funcs.append(l2_chapeu_func)

    # Calculate intensities for point
    intensities_common_axis = np.zeros((len(l2_chapeu_funcs), len(l2_chapeu_axis)))
    for col, l2_val in enumerate(l2_chapeu_axis):
        for row, l2_chapeu_func in enumerate(l2_chapeu_funcs):
            intensities_common_axis[row, col] = l2_chapeu_func(l2_val)

    return intensities_common_axis, l2_chapeu_axis


def plot_all_intensities_columns(intensities_common_axis, l2_chapeu_axis):

    """ Função que plota todas as intensidades normalizadas a partir da linha do centro

    Parâmetros:
    -----------
    intensities_common_axis: ndarray, float
        vetor que contém as intensidades normalizadas
    l2_chapeu_axis: ndarray, float
        contém as informações sobre onde o eixo deve começar e terminar. Existe uma modificação na exibição do gráfic,
        ao invés de começar do ponto de origem [0,0]. Ele (ponto de origem) vai começar dependendo da quantidade de
        linhas que existirem.
    Retorno:
    -----------
        plota todas as intensidades das colunas
    """

    # EXPLICAR MELHOR
    plt.figure(figsize=[12, 10])
    for intens in intensities_common_axis:
        plt.plot(l2_chapeu_axis, intens)

    plt.show()


def plot_fill_means_std_dev_normal_all(intensities_common_axis):
    """ Função que plota todas as intensidades normalizadas, exibindo a diferença entre a média e o desvio padrão
    existente entre as intensidades.

    Parâmetros:
    -----------
    intensities_common_axis: ndarray, float
        vetor que contém as intensidades normalizadas
    Retorno:
    -----------
        plota todas as intensidades normalizadas, exibindo a diferença entre a média e o desvio padrão
        existente entre as intensidades.
    """
    # retorna a média de todos os valores mapeados ao longo das linhas
    means = np.mean(intensities_common_axis, axis=0)

    # retorna o desvio padrão de todos os valores mapeados ao longo das linhas
    std_dev = np.std(intensities_common_axis, axis=0)

    plt.figure(figsize=[12, 10])
    plt.title("Filling between the mean intensity and standard deviation in columns")

    # mostra o sombreamento
    plt.fill_between(range(len(means)), means - std_dev, means + std_dev, alpha=0.3)

    # mostra a média
    plt.plot(range(len(means)), means)
    plt.show()


# função que plota os mínimos e máximos da linha medial de todas as extrações
def plot_min_max_medial_line(minimum, maximum):
    """ Função que plota todas os valores mínimos e máximos da linha medial de cada vaso extraído. Cada modelo de vaso
    possui um valor máximo e um mínimo de intensidade da linha medial. Esta função serve para visualizar
    estas variações.

    Parâmetros:
    -----------
    minimum: list, float
        lista contendo os valores mínimos de intensidade da linha medial de cada um dos vasos
    maximum: list, float
        lista contendo os valores máximos de intensidade da linha medial de cada um dos vasos
    Retorno:
    -----------
       plota todas os valores mínimos e máximos da linha medial de cada vaso extraído
    """

    maximum = np.array(maximum)
    minimum = np.array(minimum)
    plt.figure(figsize=[12, 10])
    plt.title(f'Max and Min of line medial:')
    plt.ylabel('Number')
    plt.xlabel('Values')
    plt.plot(minimum.flatten(), label=f'minimum')
    plt.plot(maximum.flatten(), label=f'maximum')
    plt.legend(loc='lower right')
    plt.show()


def ready_directory(diretorio):
    """ Função que lê todos os arquivos de um diretório, retornando a quantidade existente e os nomes dos arquivos

    Parâmetros:
    -----------
    diretorio: str
      nome do local onde se encontra o diretório a ser lido
    Retorno:
    -----------
    nome: list, str
      lista de nomes dos arquivos que estão sendo lidos no diretório setado
    qtde: int
      quantidade de arquivos existentes no diretório
    """

    qtde_de_arquivos = 0
    lista_de_nomes = []
    # varredura dos arquivos e adição dos nomes na variável nome e quantidade na variável qtde
    for name in os.listdir(diretorio):
        path = os.path.join(diretorio, name)
        if os.path.isfile(path):
            lista_de_nomes.append(path)
            qtde_de_arquivos += 1
    return lista_de_nomes, qtde_de_arquivos


pasta = "C:\\Users\\adria\\PycharmProjects\\pythonProject2\\dash_components\\"

diretorio = f'{pasta}Vessel_Models\\'

arquivo_json = f'{pasta}Vessel_Models\\Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 4-20X.json'

# coloca nas varíaveis nom e tam, os nomes dos arquivos que existem no diretório e pega a quantidade de itens que existem
nom, tam = ready_directory(diretorio)
print(tam)

maximum = []
minimum = []

# faz a varredura de todos os itens que estão no diretório
for i in range(1):
    # pega o nome da imagem e armazena na variável local
    local = nom[i]

    # faz a carregamento do .pickle
    data_dump = pickle.load(open(local, "rb"))
    # absorve os três índices que existem no .pickle
    vessel_model = data_dump['vessel_model']
    first_point = data_dump['primeiro_ponto']
    img_file = data_dump['img_file']

    # instanciação da variável vessel_map
    # o mapa do vaso possui vários atributos que serão utilizados para chamar as funções
    vessel_map = vessel_model.vessel_map

    # pega a metade inteira do tamanho do vessel_map.mapped_values
    half_size_vessel_map = len(vessel_map.mapped_values) // 2

    # variáveis que pegam o índice do menor e do menor valor da posição do meio
    max = np.argmax(vessel_map.mapped_values[half_size_vessel_map])
    min = np.argmin(vessel_map.mapped_values[half_size_vessel_map])

    # variáveis que pegam o valor armazenados no menor e maior valor encontrado
    maximum.append(vessel_map.mapped_values[half_size_vessel_map][max])
    minimum.append(vessel_map.mapped_values[half_size_vessel_map][min])

    # retorna a média de todos os valores mapeados ao longo das linhas
    means = np.mean(vessel_map.mapped_values, axis=1)

    # retorna o desvio padrão de todos os valores mapeados ao longo das linhas
    std_dev = np.std(vessel_map.mapped_values, axis=1)

    # retorna o desvio padrão de todos os valores mapeados ao longo das colunas
    std_dev2 = np.std(vessel_map.mapped_values, axis=0)

    # plot of the vessel map, min value is 0 and max value is 60
    plot_vessel_map(vessel_map)

    # plot do recorte
    plot_clipping(vessel_map)

    # plot da intensidade das linhas interemediária, uma acima e uma abaixo
    plot_intensity_lines(vessel_map, half_size_vessel_map)

    # plot mostrando a diferença entre a média e o desvio padrão
    plot_fill_means_std_dev(means, std_dev)

    # tentativa de plotar a diferença entre o desvio padrão com a coluna normalizada
    # plot_fill_means_std_dev_normal(intensidy_cols_values_all[0], std_dev2)

    # plota o diâmetro do vaso
    plot_diameter_vessel(vessel_map)

    # plota a intensidade das colunas, exibe onde começa e termina o vaso
    plot_intensity_cols_with_line_vessel(vessel_map)

    # plota a intensidade das colunas normalizadas com a linha do centro (metade dos índices das colunas),
    # exibe onde começa e termina o vaso, retirando a dependência de se começar do ponto (0,0)
    plot_intensity_cols_with_line_vessel_normal(vessel_map)

    # plota todas as intensidades normalizadas
    intensities_common_axis, l2_chapeu_axis = return_all_instisitys_normal(vessel_map)

    plot_all_intensities_columns(intensities_common_axis, l2_chapeu_axis)

    plot_fill_means_std_dev_normal_all(intensities_common_axis)

# plota a diferença entre os máximos e mínimos de todas as extrações
plot_min_max_medial_line(minimum, maximum)

# VER plt.xaxis.set_Visible(False)

