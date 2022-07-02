import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import json
from scipy import ndimage as ndi
import dash_components.slice_mapper_util as smutil
import dash_components.slice_mapper as slice_mapper
from dash_components.slice_mapper import SliceMapper
from scipy.ndimage import map_coordinates
from PIL import Image
from skimage import io
from numpy import random
from scipy.interpolate import interp1d

#Inverte vetor que está como coluna/linha para linha/coluna
def invertendo_linhas_colunas(vetor):
  vet=[]
  vet_aux = []
  for i in range(len(vetor)):
    vet = vetor[i]
    for j in range(len(vet)):
      vet[j] = vet[j][::-1]
    vet_aux.append(vet)
  return vet_aux


# lê um arquivo e retorna um array dos dados do arquivo
# O array está no formato coluna/linha. O código que faz a extração grava no formato coluna/linha
# Tem várias partes do vessel analysis e do slice_mapper que usam desta forma.
def retorna_paths(arq):
    # leitura do json
    q = json.load(open(arq, 'r'))

    # transforma todos os itens lidos em np.array
    path1 = [np.array(item) for item in q]

    # Função com uma linha para inverter todos os valores
    # path1 = [np.array(item)[:,::-1] for item in q]

    return path1


from scipy.ndimage.filters import median_filter


# função que pega os dois primeiros valores das linhas e colunas dos vetores passador por parâmentro e
# retorna o alcance. Calcula a distância Euclidiana entre os quatro pontos.

def setar_alcance(array_1, array_2):
    # pega a coluna 1 do vetor 1
    coluna1 = array_1[0][0]

    # pega a linha 1 do vetor 1
    linha1 = array_1[0][1]

    # pega a coluna 2 do vetor 2
    coluna2 = array_2[0][0]

    # pega a linha 2 do vetor 2
    linha2 = array_2[0][1]

    # O alcance vai ser a raiz quadrada do resultado da diferença quadrática entre os dois pontos dos dois vetores - distância Euclidiana
    # A variável alcance_extra permite que a região setada pelo alcance seja um pouco maior. O alcance é aumentado, tanto acima, como abaixo dos valores mapeados
    alcance = int(np.sqrt((linha1 - linha2) ** 2 + (coluna1 - coluna2) ** 2) + alcance_extra)

    return alcance


# função que retorna os valores mínimos e máximos de dois vetores
def retorna_linhas_colunas(caminhos):
    # pega a primeira posição do vetor
    caminho1 = caminhos[0]

    # pega a segunda posição do vetor
    caminho2 = caminhos[1]

    min_coluna1, min_linha1 = np.min(caminho1, axis=0)
    min_coluna2, min_linha2 = np.min(caminho2, axis=0)

    max_coluna1, max_linha1 = np.max(caminho1, axis=0)
    max_coluna2, max_linha2 = np.max(caminho2, axis=0)

    min_coluna = int(np.min([min_coluna1, min_coluna2]))
    min_linha = int(np.min([min_linha1, min_linha2]))
    max_coluna = int(np.max([max_coluna1, max_coluna2]))
    max_linha = int(np.max([max_linha1, max_linha2]))

    return min_linha, min_coluna, max_linha, max_coluna


# pega um vetor e uma imagem e diminui o campo da imagem
def diminui_imagem(caminhos, img_path):
    # padding setado para mostrar uma região um pouco maior do que a do vaso em questão
    padding = 5

    # retorna as menores e as maiores linhas e colunas dos caminhos
    menor_linha, menor_coluna, maior_linha, maior_coluna = retorna_linhas_colunas(caminhos)

    # pega o primeiro_ponto na posição da menor coluna, e da menor linha, decrescidos do padding
    primeiro_ponto = np.array([menor_coluna - padding, menor_linha - padding])

    # absorve os caminhos transladados decrescidos do primmeiro ponto, varrendo todo o caminho
    caminhos_transladados = [caminho - primeiro_ponto for caminho in caminhos]

    # imagem que absorve a img_path e mostra a região delimitada pelos parâmetros pelas menores e maiores linhas/colunas
    # o parâmetro "-padding" permite que seja pega uma região um pouco maior em todos os valores das linhas/colunas
    img1 = np.array(Image.open(img_path))[menor_linha - padding:maior_linha + padding,
           menor_coluna - padding:maior_coluna + padding]

    return img1, caminhos_transladados, primeiro_ponto


# função que faz o plot do mapa do vaso. Mapeia os valores de zero sendo o mínimo e 60 como sendo o máximo.
def plot_vessel_map(vessel_map):
    plt.figure(figsize=[12, 10])
    plt.title("Map values Vmin=0 e Vmax=60")
    # o mapped_values, são os valores das intensidades dos pixels do vaso sanguíneo.
    # o parâmetro alcance utilizado na função setar_alcance serve para delimitar a região que será exibida da imagem.
    plt.xticks([])
    plt.yticks([])
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)

    # os valores armazenados no path1 e path2 são as demarcações manuais feitas nos vasos

    # mostra os valores do path1 mapeado em amarelo
    plt.plot(vessel_map.path1_mapped, c='yellow')

    # mostra os valores do path2 mapeado em amarelo
    plt.plot(vessel_map.path2_mapped, c='yellow')

    plt.show()


# função que plota a intensidade das linha mediana, uma acima e uma abaixo dos valores mapeados
def plot_intensidy_lines(vessel_map, half_size_vessel_map):
    plt.figure(figsize=[12, 10])
    plt.title(
        f'Intensidy of position in sections of the vessel {half_size_vessel_map - 1} , {half_size_vessel_map} and {half_size_vessel_map + 1}')

    # up
    plt.plot(vessel_map.mapped_values[half_size_vessel_map - 1].flatten(),
             label=f'position:  {half_size_vessel_map - 1}')
    # center
    plt.plot(vessel_map.mapped_values[half_size_vessel_map].flatten(), label=f'position:  {half_size_vessel_map}')
    # down
    plt.plot(vessel_map.mapped_values[half_size_vessel_map + 1].flatten(),
             label=f'position:  {half_size_vessel_map + 1}')

    plt.legend(loc='lower right')
    plt.xlabel('Positions')
    plt.ylabel('Intensity')
    plt.show()


# função que plota a diferença entre a média e o desvio padrão
def plot_fill_means_std_dev(means, std_dev):
    plt.figure(figsize=[12, 10])
    plt.title("Filling between the mean intensity and standard deviation")

    # mostra o sombreamento
    plt.fill_between(range(len(means)), means - std_dev, means + std_dev, alpha=0.3)

    # mostra a média
    plt.plot(range(len(means)), means)
    plt.show()


# esta função só foi começada - não está terminada
def plot_fill_means_std_dev_normal(intensidy_cols_values_all, std_dev):
    plt.figure(figsize=[12, 10])
    plt.title("Filling between the intensity normalized and standard deviation")
    plt.fill_between(range(len(intensidy_cols_values_all)), intensidy_cols_values_all - std_dev,
                     intensidy_cols_values_all + std_dev, alpha=0.3)
    plt.plot(range(len(intensidy_cols_values_all)), intensidy_cols_values_all)
    plt.show()


# função que plota o diametro dos vasos mapeados
def plot_diameter_vessel(vessel_map):
    plt.figure(figsize=[12, 10])

    # o diâmetro é o módulo da difrença entre os dois caminhos mapeados.
    diameter = np.abs(vessel_map.path1_mapped - vessel_map.path2_mapped)

    plt.title("Diameter of the vessel")
    plt.xlabel('Column index')
    plt.ylabel('Diameter')

    # o diâmetro é float, portanto necessitou do range(len)
    plt.plot(range(len(diameter)), diameter)
    plt.show()


# função que armazena todas as intensidades das colunas
def return_intensidy_cols(vessel_map):
    # número de linhas e colunas do mapa do vaso
    num_rows, num_cols = vessel_map.mapped_values.shape

    intensidy_cols_values_all = []

    # armazena todas as intensidades das colunas ao longo das linhas
    for i in range(num_cols):
        intensidy_cols_values_all.append(vessel_map.mapped_values[0:num_rows, i])
    return intensidy_cols_values_all


# função que faz o recorte da imagem
def return_clipping(vessel_map):
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


# função que plota o recorte, com valores mínimos de zero e máximo de 60
def plot_clipping(vessel_map):
    # chama a função que retorna o recorte
    clipp = return_clipping(vessel_map)

    plt.figure(figsize=[8, 5])
    plt.title("Image clipping")
    plt.imshow(clipp, 'gray', vmin=0, vmax=60)
    plt.show()


# função que plota a intensidade das colunas.
# exibe onde começa e termina o vaso através das barras centrais, perperdinculares ao eixo y
def plot_intensidy_cols_with_line_vessel(vessel_map):
    array_min_path = []
    array_max_path = []

    # número de linhas e colunas dos valores mapeados
    num_rows, num_cols = vessel_map.mapped_values.shape

    # cálculo do diâmetro
    diameter = np.abs(vessel_map.path1_mapped - vessel_map.path2_mapped)

    # várias cores para alinhar a cor das colunas que serão exibidas com as v_lines que mostram a delimitação dos vasos
    colors = ['blue', 'green', 'red', 'orange', 'gray']

    # chama a função que pega todas as intensidades das colunas
    intensidy_cols_values_all = return_intensidy_cols(vessel_map)

    # Pegando a posição 0, 1/4, 1/2, 3/4, e final das colunas
    colunas_demarcadas = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]

    plt.figure(figsize=[12, 10])
    plt.title(
        f'Intensity of columns {colunas_demarcadas[0]} , {colunas_demarcadas[1]}  , {colunas_demarcadas[2]} , {colunas_demarcadas[3]} and {colunas_demarcadas[4]}')
    plt.xlabel('Line index')
    plt.ylabel('Intensidy')
    for i in range(len(colunas_demarcadas)):
        # plota as posições existentes nas colunas demarcadas no vetor que contém todas as intensidades das colunas
        plt.plot(range(num_rows), intensidy_cols_values_all[colunas_demarcadas[i]],
                 label=f'position:  {colunas_demarcadas[i]}', color=colors[i])
    plt.legend(loc='lower right')

    for j in range(len(colunas_demarcadas)):
        min_path = np.argmin(intensidy_cols_values_all[colunas_demarcadas[j]])
        array_min_path.append(intensidy_cols_values_all[colunas_demarcadas[j]][min_path])

        max_path = np.argmax(intensidy_cols_values_all[colunas_demarcadas[j]])
        array_max_path.append(intensidy_cols_values_all[colunas_demarcadas[j]][max_path])

        plt.vlines(vessel_map.path1_mapped[colunas_demarcadas[j]], array_min_path[j], array_max_path[j],
                   color=colors[j])
        plt.vlines(vessel_map.path2_mapped[colunas_demarcadas[j]], array_min_path[j], array_max_path[j],
                   color=colors[j])
    plt.show()


def plot_intensidy_cols_with_line_vessel_normal(vessel_map, colunas_demarcadas=None):
    num_rows, num_cols = vessel_map.mapped_values.shape

    if (colunas_demarcadas is None):
        # Mostrando a posição 0, 1/4, 1/2, 3/4, e final das colunas
        colunas_demarcadas = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]

    # recebe um vetor de cores
    colors = ['blue', 'green', 'red', 'orange', 'gray']

    # puxa todas as intensidades de todas as colunas
    intensidy_cols_values_all = return_intensidy_cols(vessel_map)

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
        plt.plot(l2_chapeu_all[i], intensidy_cols_values_all[colunas_demarcadas[i]], '-o',
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
        min_path = np.argmin(intensidy_cols_values_all[colunas_demarcadas[i]])
        array_min_path.append(intensidy_cols_values_all[colunas_demarcadas[i]][min_path])

        max_path = np.argmax(intensidy_cols_values_all[colunas_demarcadas[i]])
        array_max_path.append(intensidy_cols_values_all[colunas_demarcadas[i]][max_path])
    plt.vlines(liv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors)
    plt.vlines(lfv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors)

    # VER
    plt.xlabel('Positions')
    plt.ylabel('Intensity')

    plt.legend(loc='lower right')
    plt.show()


def return_all_instisitys_normal(vessel_map):
    vessel_map = vessel_model.vessel_map
    num_rows, num_cols = vessel_map.mapped_values.shape

    # puxa todas as intensidades de todas as colunas
    intensidy_cols_values_all = return_intensidy_cols(vessel_map)

    # Mostrando a posição 0, 1/4, 1/2, 3/4, e final das colunas
    colunas_demarcadas = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]

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
    for col in range(len(intensidy_cols_values_all)):
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
    for l2, intens in zip(l_all, intensidy_cols_values_all):
        l2_chapeu_func = interp1d(l2, intens, kind='linear', bounds_error=False)
        l2_chapeu_funcs.append(l2_chapeu_func)

    # Calculate intensities for point
    intensities_common_axis = np.zeros((len(l2_chapeu_funcs), len(l2_chapeu_axis)))
    for col, l2_val in enumerate(l2_chapeu_axis):
        for row, l2_chapeu_func in enumerate(l2_chapeu_funcs):
            intensities_common_axis[row, col] = l2_chapeu_func(l2_val)

    return intensities_common_axis, l2_chapeu_axis


def plot_all_intensities_columns(intensities_common_axis, l2_chapeu_axis):
    plt.figure(figsize=[12, 10])
    for intens in intensities_common_axis:
        plt.plot(l2_chapeu_axis, intens)

    plt.show()


def plot_fill_means_std_dev_normal_all(intensities_common_axis):
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


# função que lê todos os arquivos de um diretório, retornando a quantidade existente e os nomes dos arquivos
def ready_directory(dir):
    qtde = 0
    nom = []
    # varredura dos arquivos e adição dos nomes na variável nom e quantidade na variável qtde
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isfile(path):
            nom.append(path)
            qtde = qtde + 1
    return nom, qtde


# BACKUP
def return_intensidy_cols2(vessel_map):
    num_rows, num_cols = vessel_map.mapped_values.shape
    colunas_demarcadas = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]
    intensidy_cols_values = []
    intensidy_cols_values_all = []
    for i in range(5):
        col = colunas_demarcadas[i]
        intensidy_cols_values.append(vessel_map.mapped_values[0:num_rows, col])

        # acrescentei para colocar todas as intensidades das colunas ao longo das linhas
    for i in range(num_cols):
        intensidy_cols_values_all.append(vessel_map.mapped_values[0:num_rows, i])

        # retorna as intensidades específicas, todas as intensidades e as colunas demarcadas
    return intensidy_cols_values, intensidy_cols_values_all, colunas_demarcadas


# plota as imagens
def return_vessel(img, p1, p2, reach):
    vessel_model, cross_paths = slice_mapper.map_slices(img, p1, p2, delta_eval, smoothing, reach)

    # vessel_model10, cross = slice_mapper.map_slices(img, caminhos_transladados[0], caminhos_transladados[1], delta_eval, smoothing,  reach)
    # cross_paths = slice_mapper.create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=True)
    # vessel_model = slice_mapper.map_slices(img, p1, p2, delta_eval, smoothing, reach)

    return vessel_model, cross_paths


def plot_figure(img, vessel_model, cross):
    vessel_map = vessel_model.vessel_map
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot()
    slice_mapper.plot_model(img, vessel_model, cross, ax)
    plt.title("Imagem analisada")
    norm = ax.images[0].norm
    norm.vmin, norm.vmax = 0, 60
    plt.figure(figsize=[8, 5])
    plt.title("Vmin=0 e Vmax=60")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)

    # plt.plot(vessel_map.path1_mapped, c='green')
    # plt.plot(vessel_map.path2_mapped, c='green')

    plt.figure(figsize=[8, 5])
    plt.title("Vmin=0 e Vmax=255")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=255)
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')

    plt.figure(figsize=[8, 5])
    plt.title("Vmin=0 e Vmax=maximo valor mapeado")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=vessel_map.mapped_values.max())
    # skimage.io.imsave('teste1.jpg', vessel_map.mapped_values[::-1], plugin=None, check_contrast=True,  vmin=0, vmax=60)

    # Gravação da imagem pegando o vmin e vmax como parâmetro
    # plt.imsave('teste2.jpg', vessel_map.mapped_values, cmap='gray', vmin=0, vmax=60)
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')


smoothing = 0.01
delta_eval = 1.
# reach = 35.
alcance_extra = 6

imag = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 4-20X'



arquivo = f'{imag}.json'

path = f'{imag}.tiff'

# pega o arquivo e armazena em um array
array_path = retorna_paths(arquivo)

# leitura da imagem
img = np.array(Image.open(path))

# pega a metade inteira do vetor
half_array = len(array_path) // 2

x = 0
for i in range(half_array):
    img, caminhos_transladados, primeiro_ponto = diminui_imagem(array_path[x:x + 2], path)

    # cam_invertido = invertendo_linhas_colunas(caminhos_transladados)

    # vessel_model = plot_figure(img, cam_invertido[0], cam_invertido[1], setar_alcance(array_path[0], array_path[1]))
    # vessel_model = plot_figure(img, caminhos_transladados[0], caminhos_transladados[1], setar_alcance(array_path[0], array_path[1]))
    vessel_mod, cross_t = return_vessel(img, caminhos_transladados[0], caminhos_transladados[1],
                                        setar_alcance(array_path[0], array_path[1]))
    plot_figure(img, vessel_mod, cross_t)
    # data_dump = {"img_file": path, "vessel_model": vessel_mod, "primeiro_ponto": primeiro_ponto}
    # savedata = f'{pasta_mestrado}/Vessel_Models/{imag}_savedata{i}.pickle'
    # pickle.dump(data_dump, open(savedata,"wb"))
    x += 2