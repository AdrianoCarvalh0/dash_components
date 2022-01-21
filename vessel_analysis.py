import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
from scipy import ndimage as ndi
import dash_components.slice_mapper_util
import dash_components.slice_mapper as slice_mapper
from dash_components.slice_mapper import SliceMapper
from scipy.ndimage import map_coordinates
from PIL import Image

from skimage import io

def invertendo_linhas_colunas(vetor):
  vet=[]
  vet_aux = []
  for i in range(len(vetor)):
    vet = vetor[i]
    for j in range(len(vet)):
      vet[j] = vet[j][::-1]
    vet_aux.append(vet)
  return vet_aux

#lê um arquivo e retorna um array dos dados do arquivo
def retorna_paths(arq):

  q = json.load(open(arq, 'r'))
  path1 = [np.array(item) for item in q]
  #path1 = [np.array(item)[:,::-1] for item in q]
  #ar_path  = invertendo_linhas_colunas(path1)
  return path1



def setar_alcance(array_1, array_2):
    coluna1 = array_1[0][0]
    linha1 = array_1[0][1]
    coluna2 = array_2[0][0]
    linha2 = array_2[0][1]
    alcance = int(np.sqrt((linha1 - linha2) ** 2 + (coluna1 - coluna2) ** 2) + alcance_extra)
    return alcance


def retorna_linhas_colunas(caminhos):
    caminho1 = caminhos[0]
    caminho2 = caminhos[1]
    # min_linha1, min_coluna1 = np.min(caminho1,axis=0)
    # min_linha2, min_coluna2 = np.min(caminho2,axis=0)

    # max_linha1, max_coluna1 = np.max(caminho1,axis=0)
    # max_linha2, max_coluna2 = np.max(caminho2,axis=0)

    min_coluna1, min_linha1 = np.min(caminho1, axis=0)
    min_coluna2, min_linha2 = np.min(caminho2, axis=0)

    max_coluna1, max_linha1 = np.max(caminho1, axis=0)
    max_coluna2, max_linha2 = np.max(caminho2, axis=0)

    min_coluna = int(min([min_coluna1, min_coluna2]))
    min_linha = int(min([min_linha1, min_linha2]))
    max_coluna = int(max([max_coluna1, max_coluna2]))
    max_linha = int(max([max_linha1, max_linha2]))

    return min_linha, min_coluna, max_linha, max_coluna


# pega um vetor e uma imagem e diminui o campo da imagem
def diminui_imagem(caminhos, img_path):
    padding = 5

    menor_linha, menor_coluna, maior_linha, maior_coluna = retorna_linhas_colunas(caminhos)
    # print(menor_linha, menor_coluna, maior_linha, maior_coluna)

    # Era assim que estava primeiro_ponto = np.array([menor_coluna-padding,  menor_linha-padding])
    primeiro_ponto = np.array([menor_coluna - padding, menor_linha - padding])

    caminhos_transladados = [caminho - primeiro_ponto for caminho in caminhos]

    # print(caminhos_transladados)

    img1 = np.array(Image.open(img_path))[menor_linha - padding:maior_linha + padding,
           menor_coluna - padding:maior_coluna + padding]

    return img1, caminhos_transladados, primeiro_ponto


# plota as imagens
def return_vessel(img, p1, p2, reach):
    vessel_model = slice_mapper.map_slices(img, p1, p2, delta_eval, smoothing, reach)
    cross_paths = slice_mapper.create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=True)
    # vessel_model = slice_mapper.map_slices(img, p1, p2, delta_eval, smoothing, reach)
    vessel_map = vessel_model.vessel_map
    return vessel_model, vessel_map, cross_paths


def plot_figure(img, vessel_map, vessel_model, cross_paths):
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot()
    slice_mapper.plot_model(img, vessel_model, cross_paths, ax)
    plt.title("Imagem analisada")
    norm = ax.images[0].norm
    norm.vmin, norm.vmax = 0, 60
    plt.figure(figsize=[8, 5])
    plt.title("Vmin=0 e Vmax=60")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)

    plt.plot(vessel_map.path1_mapped, c='red')
    plt.plot(vessel_map.path2_mapped, c='red')

    plt.figure(figsize=[8, 5])
    plt.title("Vmin=0 e Vmax=255")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=255)
    plt.plot(vessel_map.path1_mapped, c='yellow')
    plt.plot(vessel_map.path2_mapped, c='yellow')

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

pasta_mestrado = '/content/drive/MyDrive/Mestrado em Ciência da Computação/'

arquivo = f'{pasta_mestrado}/Vetores extraidos/{imag}.json'

path = f'{pasta_mestrado}Imagens/vessel_data/maiores/{imag}.tiff'

# pega o arquivo e armazena em um array
array_path = retorna_paths(arquivo)

# leitura da imagem
# img = np.array(Image.open(path))


# pega a metade inteira do vetor
half_array = len(array_path) // 2

x = 0
array_pickle = []
for i in range(half_array):
    img, caminhos_transladados, primeiro_ponto = diminui_imagem(array_path[x:x + 2], path)

    # cam_invertido = invertendo_linhas_colunas(caminhos_transladados)

    # vessel_model = plot_figure(img, cam_invertido[0], cam_invertido[1], setar_alcance(array_path[0], array_path[1]))
    # vessel_model = plot_figure(img, caminhos_transladados[0], caminhos_transladados[1], setar_alcance(array_path[0], array_path[1]))
    vessel_model, vessel_map, cross_paths = return_vessel(img, caminhos_transladados[0], caminhos_transladados[1],
                                                          setar_alcance(array_path[0], array_path[1]))
    plot_figure(img, vessel_map, vessel_model, cross_paths)  # , vessel_model)
    data_dump = {"img_file": path, "vessel_model": vessel_model, "primeiro_ponto": primeiro_ponto}
    savedata = f'{pasta_mestrado}{imag}/savedata{i}.pickle'
    pickle.dump(data_dump, open(savedata, "wb"))
    x += 2

for i in range(3):
  localiza = f'{pasta_mestrado}{imag}/savedata{i}.pickle'
  data_dump = pickle.load(open(localiza,"rb"))
  vessel_model = data_dump['vessel_model']
  primeiro_ponto = data_dump['primeiro_ponto']
  img_file = data_dump['img_file']
  vessel_map = vessel_model.vessel_map

  plt.figure(figsize=[8, 5])
  plt.title("Vmin=0 e Vmax=255")
  plt.plot()
  plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=255)
  plt.plot(vessel_map.path1_mapped, c='yellow')
  plt.plot(vessel_map.path2_mapped, c='yellow')