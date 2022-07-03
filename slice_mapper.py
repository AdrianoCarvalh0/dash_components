import numpy as np
from matplotlib.patches import Arrow, ArrowStyle, FancyArrow
from matplotlib.collections import PatchCollection
from scipy.ndimage import map_coordinates
import dash_components.slice_mapper_util as smutil
from shapely import geometry, ops as shops, affinity
from IPython.display import display
from skimage import draw


# criação de classes para facilitar o encapsulamento das variáveis, atributos e funções.
class SliceMapper:
    """Classe que dá origem ao modelo e ao mapa do vaso. Chama as funções que criam o VesselModel e VesselMap

        Parâmetros:
        -----------
        img: ndarray, float
            imagem original
        delta_eval: float
            parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
        smoothing: float
            critério de suavização
        reach: float
            variável que delimita o tamanho do mapa do vaso. seta o alcance superior e inferior que o
            mapa irá abranger
        add_model:
        -----------
            cria o modelo e o mapa do vaso
        """

    # comentar os atributos
    def __init__(self, img, delta_eval, smoothing, reach):
        self.img = img
        self.delta_eval = delta_eval
        self.smoothing = smoothing
        self.reach = reach
        self.models = []
        self.debug = []

    def add_model(self, path1, path2, generate_map=True):
        vessel_model = create_vessel_model(self.img, path1, path2, self.delta_eval, self.smoothing)

        if generate_map:
            vessel_map = create_map(self.img, vessel_model, self.reach,
                                    self.delta_eval, self.smoothing)
            vessel_model.set_map(vessel_map)

        self.models.append(vessel_model)


class VesselModel:
    """Classe que armazena informações relacionadas ao modelo do vaso

    Parâmetros:
    -----------
    path1: ndarray, float
        vetor do caminho 1
    path1_info: tuple
       informações sobre o caminho 1 são armazenadas em vetores numpy e armazenadas no path1_info
    path2: ndarray, float
        vetor do caminho 2
    path2_info: tuple
       informações sobre o caminho 2 são armazenadas em vetores numpy e armazenadas no path2_info
    medial_path: ndarray, float
        caminho medial
    medial_info: ndarray, float
         informações sobre a linha medial são armazenadas em vetores numpy e armazenadas no medial_info
    delta_eval: float
        parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    vessel_map: object VesselMap
        instância do objeto VesselMap
    img_file: ndarray, float
        arquivo da imagem
    Retorno:
    -----------
        absorve as informações passadas no construtor e armazena no objeto VesselModel
    """

    def __init__(self, path1, path1_info, path2, path2_info, medial_path, medial_info,
                 delta_eval, vessel_map=None, img_file=None):
        self.path1 = {
            'original': path1,
            'interpolated': path1_info[0],
            'tangents': path1_info[1],
            'normals': path1_info[2],
        }

        self.path2 = {
            'original': path2,
            'interpolated': path2_info[0],
            'tangents': path2_info[1],
            'normals': path2_info[2],
        }

        self.medial_path = {
            'original': medial_path,
            'interpolated': medial_info[0],
            'tangents': medial_info[1],
            'normals': medial_info[2],
        }
        self.delta_eval = delta_eval
        self.vessel_map = vessel_map
        self.img_file = img_file

    def set_map(self, vessel_map):
        self.vessel_map = vessel_map


class VesselMap:
    """Classe que armazena as informações relacionadas ao mapa do vaso

       Parâmetros:
       -----------
       mapped_values: ndarray, float
           valores mapeados
       medial_coord: ndarray, float
          coordenadas mediais
       cross_coord: ndarray, float
          coordenadas transversais
       cross_versors: list, float
          lista contendo os versores transversais
       mapped_mask_values: ndarray, float
           valores mapeados em binário
       path1_mapped: ndarray, float
           caminho 1 mapeado
       path2_mapped: ndarray, float
           caminho 2 mapeado
       Retorno:
       -----------
           absorve as informações passadas no construtor e armazena no objeto VesselMap
       """

    def __init__(self, mapped_values, medial_coord, cross_coord, cross_versors, mapped_mask_values,
                 path1_mapped, path2_mapped):
        self.mapped_values = mapped_values
        self.medial_coord = medial_coord
        self.cross_coord = cross_coord
        self.cross_versors = cross_versors
        self.mapped_mask_values = mapped_mask_values
        self.path1_mapped = path1_mapped
        self.path2_mapped = path2_mapped


def interpolate_envelop(path1, path2, delta_eval=2., smoothing=0.01):
    """Envelopa os itens, caminhos1 e caminho2, suas interpolações suavizadas, suas tangentes e suas normais.
    
    Parâmetros:
    -----------
    path1: ndarray, float
        vetor do caminho 
    path2: ndarray, float
        vetor do caminho 
    delta_eval: float
        parâmento que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    smoothing: float
        critério de suavização
    Retorno
    -----------  
    path1: ndarray, float
        caminho 1
    path1_interp: ndarray, float
        caminho 1 interpolado e suavizado
    tangents1: ndarray, float
        vetor de tangentes do caminho 1
    normals1: ndarray, float
        vetor de normais do caminho 1
    path2: ndarray, float
        caminho 2
    path2_interp: ndarray, float
        caminho 2 interpolado e suavizado
    tangents2: ndarray, float
        vetor de tangentes do caminho 2
    normals2: ndarray, float
       vetor de normais do caminho 2
    """

    # os caminhos são interpolados e novas tangentes são criadas a partir da interpolação dos caminhos
    path1_interp, tangents1 = smutil.two_stage_interpolate(path1, delta_eval=delta_eval, smoothing=smoothing)
    path2_interp, tangents2 = smutil.two_stage_interpolate(path2, delta_eval=delta_eval, smoothing=smoothing)

    # vetores normais são criados a partir das novas tangentes
    normals1 = smutil.get_normals(tangents1)
    normals2 = smutil.get_normals(tangents2)

    min_size = min([len(path1_interp), len(path2_interp)])

    # Faz as normais apontarem para direções opostas.
    congruence = np.sum(np.sum(normals1[:min_size] * normals2[:min_size], axis=1))
    if congruence > 0:
        normals2 *= -1

    # Faz as normais apontarem para o interior
    vsl1l2 = path2_interp[:min_size] - path1_interp[:min_size]
    congruence = np.sum(np.sum(vsl1l2 * normals1[:min_size], axis=1))
    if congruence < 0:
        normals1 *= -1
        normals2 *= -1

    if np.cross(tangents1[1], normals1[1]) < 0:
        # Faz o caminho1 ser executado à esquerda do caminho2
        path1, path2 = path2, path1
        path1_interp, path2_interp = path2_interp, path1_interp
        tangents1, tangents2 = tangents2, tangents1
        normals1, normals2 = normals2, normals1

    return path1, (path1_interp, tangents1, normals1), path2, (path2_interp, tangents2, normals2)


def extract_medial_path(path1_interp, path2_interp, delta_eval=2., smoothing=0.01, return_voronoi=False):
    """Extrai o caminho medial a partir de uma estrutura tubular.

    Parâmetros:
    -----------
    path1_interp: ndarray, float
        caminho 1 interpolado
    path2_interp: ndarray, float
        caminho 2 interpolado
    delta_eval: float
        parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    smoothing: float
        critério de suavização
    return_voronoi: boolean
        quando True retona informações do objeto Voronoi criado

    Retorno
    -----------
    medial_path: ndarray, float
        caminho medial
    medial_path_info: ndarray, float
        contém o caminho medial, suas tangentes e suas normais
    vor: objeto do tipo Voronoi
        retorna informações sobre o objeto Voronoi
    """
    vor, idx_medial_vertices, point_relation = smutil.medial_voronoi_ridges(path1_interp, path2_interp)
    idx_medial_vertices_ordered = smutil.order_ridge_vertices(idx_medial_vertices)
    medial_path = []
    for idx_vertex in idx_medial_vertices_ordered:
        medial_path.append(vor.vertices[idx_vertex])
    medial_path = np.array(medial_path)
    medial_path = smutil.invert_if_oposite(path1_interp, medial_path)

    # Garante que o caminho medial vai até o final do tubo
    # tira a média dos caminhos interpolados
    first_point = (path1_interp[0] + path2_interp[0]) / 2
    last_point = (path1_interp[-1] + path2_interp[-1]) / 2
    medial_path = np.array([first_point.tolist()] + medial_path.tolist() + [last_point.tolist()])

    # interpola o caminho medial para fazer uma suavização
    medial_path_info = interpolate_medial_path(medial_path, delta_eval=delta_eval, smoothing=smoothing)

    if return_voronoi:
        return medial_path, medial_path_info, vor
    else:
        return medial_path, medial_path_info


# NÃO ESTÁ SENDO CHAMADA EM LUGAR ALGUM
def create_cross_paths_limit(path, normals, cross_coord, remove_endpoints=True):
    """Esta função cria os limites das trajetórias transversais.

    Parâmetros:
    -----------
    path: ndarray, float
        caminho
    normals: ndarray, float
        vetor contendo as normaiscros
    cross_coord: ndarray, float
        coordenadas transversais criadas a partir de uma altura, de um delta_eval e concatenadas em um arranjo
    remove_endpoints: boolean
        quando True remove os endpoints

    Retorno
    -----------
    limits: ndarray, float
        retorna os limites
    """

    if remove_endpoints:
        # É útil remover endpoints se o caminho foi interpolado
        path = path[1:-1]
        normals = normals[1:-1]

    cross_coord = cross_coord[None].T

    limits = []
    first_cross_path = path[0] + cross_coord * normals[0]
    limits.append(first_cross_path)
    cross_coord_first_p = cross_coord[0]
    cross_coord_last_p = cross_coord[-1]
    first_points = []
    last_points = []
    for point_idx, point in enumerate(path[1:-1], start=1):
        first_points.append(point + cross_coord_first_p * normals[point_idx])
        last_points.append(point + cross_coord_last_p * normals[point_idx])
    limits.append(np.array(last_points))
    last_cross_path = path[-1] + cross_coord * normals[-1]
    limits.append(last_cross_path[::-1])
    limits.append(np.array(first_points[::-1]))
    limits = np.concatenate(limits, axis=0)

    return limits


def create_vessel_model(img, path1, path2, delta_eval, smoothing):
    """Esta função cria o modelo do vaso

    Parâmetros:
    -----------
    img: ndarray, float
        imagem que dá origem à criação do modelo do vaso
    path1: ndarray, float
        caminho 1
    path2: ndarray, float
        caminho 2
    delta_eval: float
        parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    smoothing: float
        critério de suavização
    Retorno
    -----------
    vm: obejct VesselModel
        retorna o modelo do vaso com um objeto instanciado da classe VesselModel
    """

    # chama a função de inversão. Se o caminho estiver invertido o caminho2 é invertido
    path2 = smutil.invert_if_oposite(path1, path2)

    # variáveis absorvem o resultado do envelopamento de caminho1, caminho2, passamos um delta_eval 
    # que aumenta a resolução e um grau de suavização é aplicado
    path1, path1_info, path2, path2_info = interpolate_envelop(path1, path2, delta_eval, smoothing)

    # As informações contidas nos caminhos 1 e 2 são inseridas nas variáveis
    path1_interp, tangents1, normals1 = path1_info
    path2_interp, tangents2, normals2 = path2_info

    # A linha medial, juntamente com suas informações são criadas
    medial_path, medial_path_info = extract_medial_path(path1_interp, path2_interp, delta_eval=delta_eval,
                                                        smoothing=smoothing)

    # o modelo vaso é criado e passado como retorno da função
    # instanciação da classe VesselModel contendo o caimnho 1, informações sobre o caminho 1, caminho 2, informações sobre o caminho 2,
    # o caminho medial, as informações do caminho medial, e o delta_eval
    vm = VesselModel(path1, path1_info, path2, path2_info, medial_path, medial_path_info, delta_eval)

    return vm


def create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=False):
    """Cria uma imagem contendo intensidades de seção transversal ao longo do caminho medial fornecido

    Parâmetros:
    -----------
    img: ndarray, float
        imagem que dá origem à criação do mapa
    vessel_model: object VesselModel
        objeto do tipo VesselModel
    reach: float
        variável que define o quanto de limite superior e inferior a imagem terá, tem implicação direta com a quantidade de linhas do mapa criado
    delta_eval: float
        parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    smoothing: float
        critério de suavização
    return_cross_paths: boolean
        Por padrão vem False. Se True retorna os caminhos tranversais válidos
    Retorno
    -----------
    vesselmap: obejct VesselMap
        retorna o mapa do vaso como um objeto instanciado da classe VesselMap
    cross_paths_valid: ndarray
        retorna os caminhos transversais válidos
    """

    # os caminhos absorvem os valores do modelo do vaso no índice 'interpolated'
    path1_interp = vessel_model.path1['interpolated']
    path2_interp = vessel_model.path2['interpolated']

    # o caminho medial interpolado e as mediais normais são criados a partir do modelo do vaso
    medial_path_interp, medial_normals = vessel_model.medial_path['interpolated'], vessel_model.medial_path['normals']

    # as coordenadas tranversais são criadas a partir do reach (altura) e do delta_eval, concatenando os valores em um arranjo 
    cross_coord = np.concatenate((np.arange(-reach, 0 + 0.5 * delta_eval, delta_eval),
                                  np.arange(delta_eval, reach + 0.5 * delta_eval, delta_eval)))

    # os caminhos transversais e os versores tranversais são criados a partir da função de criação de caminhos transversais
    cross_paths, cross_versors = create_cross_paths(cross_coord, medial_path_interp, medial_normals, path1_interp,
                                                    path2_interp, reach)

    # a coordenada medial é criada através da suavização do comprimento do arco do caminho medial interpolado                                                    
    medial_coord = smutil.arc_length(medial_path_interp)

    cross_paths_valid = []

    # função que pega todo o caminho cruzado, verifica se está vazio e adiciona os valores válidos em um vetor
    # de caminhos cruzados válidos
    for idx, cross_path in enumerate(cross_paths[1:-1], start=1):
        if cross_path is not None:
            cross_paths_valid.append(cross_path)
    cross_paths_valid = np.array(cross_paths_valid)

    # variável que absorve os caminhos cruzados planos a partir dos pontos no caminho transversal
    cross_paths_flat = np.array([point for cross_path in cross_paths_valid for point in cross_path])

    # mapeamento dos valores são calculados a partir do método map_coordinates do scipy.ndimage, passando alguns parâmetros
    # e os caminhos tranversais planos transpostos
    mapped_values = map_coordinates(img.astype(float), cross_paths_flat.T[::-1], output=np.float, mode='mirror')

    # os caminhos mapeados são reformulados e transpostos
    mapped_values = mapped_values.reshape(-1, len(cross_coord)).T

    # geração de uma máscara para a imagem e para os valores mapeados
    # vai substituir a imagem original por uma imagem binária contendo somente o vaso
    mask_img = generate_mask(path1_interp, path2_interp, img.shape)

    # os valores binários mapeados são criados
    mapped_mask_values = map_coordinates(mask_img, cross_paths_flat.T[::-1], output=np.uint8,
                                         order=0, mode='mirror')
    mapped_mask_values = mapped_mask_values.reshape(-1, len(cross_coord)).T

    # pega as precisas posições para o caminho1 e caminho2 interpolado no mapa 
    path1_mapped, path2_mapped = find_vessel_bounds_in_map(path1_interp,
                                                           path2_interp, cross_paths_valid, delta_eval, smoothing)

    # instanciação do objeto do tipo VesselMap, armazenando os valores mapeados, as coordenadas mediais, as coordenadas transversais,
    # os versores transversais, os valores mapeados binários, o caminho 1 e 2 mapeados
    vessel_map = VesselMap(mapped_values, medial_coord, cross_coord, cross_versors, mapped_mask_values, path1_mapped,
                           path2_mapped)
    if return_cross_paths:
        return vessel_map, cross_paths_valid
    else:
        return vessel_map


def find_vessel_bounds_in_map(path1_interp, path2_interp, cross_paths, delta_eval, smoothing):
    """Encontra os limites dos vasos no mapa

    Parâmetros:
    -----------
    path1_interp: ndarray, float
        caminho 1 interpolado
    path2_interp: ndarray, float
        caminho 2 interpolado
    cross_paths: ndarray
        vetor que contém os caminhos transversais
    delta_eval: float
        parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    smoothing: float
        critério de suavização
    Retorno
    -----------
    path1_mapped: list, float
        lista que contém o mapeamento do caminho 1
    path2_mapped: list, float
        lista que contém o mapeamento do caminho 2
    """

    # LineString: O objeto LineString construído representa um ou mais splines lineares conectados entre os pontos. 
    # Pontos repetidos na sequência ordenada são permitidos, mas podem incorrer em penalidades de desempenho e 
    # devem ser evitados. Uma LineString pode se cruzar, ou seja, ser complexa e não simples.
    sh_path1_interp = geometry.LineString(path1_interp)
    sh_path2_interp = geometry.LineString(path2_interp)
    path1_mapped = []
    path2_mapped = []

    # varre os caminhos transversais
    for cross_path in cross_paths:

        # aplica o LineString no caminho transversal
        sh_cross_path = geometry.LineString(cross_path)

        # limite do caminho é obtido através das interseções dos caminhos cruzados
        path_lim = find_envelop_cross_path_intersection(sh_cross_path, sh_path1_interp)
        if path_lim is None:
            path1_mapped.append(np.nan)
        else:
            # sh_path1_cross_coord recebe o retorno da distância ao longo deste objeto geométrico até um ponto mais próximo do outro objeto.
            sh_path1_cross_coord = sh_cross_path.project(path_lim)
            path1_mapped.append(np.array(sh_path1_cross_coord))
        path_lim = find_envelop_cross_path_intersection(sh_cross_path, sh_path2_interp)

        # o mesmo procedimento é feito para o caminho2
        if path_lim is None:
            path2_mapped.append(np.nan)
        else:
            sh_path2_cross_coord = sh_cross_path.project(path_lim)
            path2_mapped.append(np.array(sh_path2_cross_coord))

    # quanto menor for o valor de delta_eval maior será a quantidade de valores mapeados
    path1_mapped = np.array(path1_mapped) / delta_eval
    path2_mapped = np.array(path2_mapped) / delta_eval

    # retorno das listas contendo os valores do caminho 1 e 2 mapeados
    return path1_mapped, path2_mapped


def find_envelop_cross_path_intersection(sh_cross_path, sh_path_interp, max_dist_factor=2.):
    """Encontra interseções dos caminhos transversais do envelope

    Parâmetros:
    -----------
    sh_cross_path: object, LineString
        objeto construído a partir da classe shapely.geometry.linestring.LineString
    sh_path_interp: object, LineString
        objeto construído a partir da classe shapely.geometry.linestring.LineString
    max_dist_factor: float
        parâmetro que define qual será o fator da maior distância
    Retorno
    -----------
    path_lim: object, Point
        objeto construído a partir da classe shapely.geometry.point.Point
    """

    # pega o índice inteiro do meio do tamanho de sh_cross_path.coords
    idx_middle_cross_point = len(sh_cross_path.coords) // 2

    # o limite do caminho é obtido através das interseções do sh_cross_path
    path_lim = sh_path_interp.intersection(sh_cross_path)
    if path_lim.is_empty:
        # Nos pontos finais, os caminhos podem não se cruzar
        path_lim = None
    else:
        sh_middle_cross_point = geometry.Point(sh_cross_path.coords[idx_middle_cross_point])
        if path_lim.geom_type == 'MultiPoint':
            # Os caminhos se cruzam em mais de um ponto, é necessário encontrar o ponto mais próximo do meio
            distances = []
            for point in path_lim:
                distances.append(sh_middle_cross_point.distance(point))
            path_lim = path_lim[np.argmin(distances)]

        min_distance = sh_middle_cross_point.distance(sh_path_interp)
        distance_path_lim = sh_middle_cross_point.distance(path_lim)
        if distance_path_lim > max_dist_factor * min_distance:
            path_lim = None

    # retorna o limite do caminho
    return path_lim


def map_slices(img, path1, path2, delta_eval, smoothing, reach):
    """Criando os modelos e mapas dos vasos

    Parâmetros:
    -----------
    path1: ndarray, float
        vetor do caminho 1
    path2: ndarray, float
        vetor do caminho 2
    delta_eval: float
        parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    smoothing: float
            critério de suavização
    reach: float
        variável que delimita o tamanho do mapa do vaso. seta o alcance superior e inferior que o
        mapa irá abranger
    Retorno:
    -----------
    vessel_model: obejct VesselModel
        retorna o modelo do vaso com um objeto instanciado da classe VesselModel
    cross_paths: ndarray, float
        caminhos transversais
    """

    # criação do modelo do vaso
    vessel_model = create_vessel_model(img, path1, path2, delta_eval, smoothing)

    # criação do mapa do vaso e dos caminhos transversais
    vessel_map, cross_paths = create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=True)
    vessel_model.set_map(vessel_map)

    # retornando o modelo do vaso e os caminhos transversais
    return vessel_model, cross_paths


def interpolate_medial_path(path, delta_eval=2., smoothing=0.01):
    """Interpolando o caminho medial

    Parâmetros:
    -----------
    path: ndarray, float
        vetor do caminho
    delta_eval: float
        parâmetro que aumenta a resolução e cria pontos intermédiários entre uma coordenada e outra (interpola)
    smoothing: float
        critério de suavização
    Retorno:
    -----------
    path_interp: ndarray, float
        caminho interpolado
    tangents: ndarray, float
        vetor contendo as tangentes
    normals: ndarray, float
        vetor contendo as normais
    """

    # o caminho interpolado e as tangentes são calculadas a partir de dois estágios de interpolação
    # o primeiro estágio é linear e o segundo é cúbico
    path_interp, tangents = smutil.two_stage_interpolate(path, delta_eval=delta_eval, smoothing=smoothing)

    # as normais são obtidas a partir das tangentes
    normals = smutil.get_normals(tangents)
    if np.cross(tangents[0], normals[0]) > 0:
        # Fazendo as normais apontarem para a "esquerda" do medial_path
        normals *= -1

    # retornando o caminho interpolado, as tangentes e as normais
    return path_interp, tangents, normals


def show_interpolated(path_interp, tangents, normals, ax, scale=2., color='blue'):
    """Mostra o caminho interpolado, juntamente com tangentes e normais. A escala passada por parâmetro define o comprimento
    das setas.

    Parâmetros:
    -----------
    path_interp: ndarray, float
        caminho interpolado
    tangents: ndarray, float
        vetor contendo as tangentes
    normals: ndarray, float
        vetor contendo as normais
    ax: object,, AxesSubplot
        objeto do tipo AxesSubplot que faz parte da biblioteca matplotlib
    scale: float
        parâmetro que dita a escala que será seguida
    color: str
        string que armazena a cor da exibição do caminho interpolado.
    Retorno:
    -----------
        os caminhos interpoloados serão plotados e serão adicionadas as colunas tangentes e normais aos eixos.
    """

    # definição do comprimento das cabeças das tangentes e das normais
    tangent_heads = path_interp + scale * tangents
    normals_heads = path_interp + scale * normals

    # estilo da seta
    arrow_style = ArrowStyle("->", head_length=10, head_width=3)

    # vetor de setas tangentes
    tangent_arrows = []
    for idx in range(len(path_interp)):
        # fa recebe o método FancyArrow, que faz parte da biblioteca do matplot.lib, que ao passar um polígono 
        # como parâmetro cria-se uma flecha (seta)
        fa = FancyArrow(path_interp[idx, 0], path_interp[idx, 1], scale * tangents[idx, 0], scale * tangents[idx, 1],
                        width=0.01, head_width=0.1, head_length=0.2, color='orange')
        # o vetor de setas tangentes é incrementado
        tangent_arrows.append(fa)
    tangents_col = PatchCollection(tangent_arrows, match_original=True, label='Tangent')

    # vetor de setas normais
    normal_arrows = []
    for idx in range(len(path_interp)):
        # fa recebe o método FancyArrow, que faz parte da biblioteca do matplot.lib, que ao passar um polígono 
        # como parâmetro cria-se uma flecha (seta)  
        fa = FancyArrow(path_interp[idx, 0], path_interp[idx, 1], scale * normals[idx, 0], scale * normals[idx, 1],
                        width=0.01, head_width=0.1, head_length=0.2, color='orange')
        normal_arrows.append(fa)
    # o PatchCollection do matplotlib pega as setas normais e adiciona no normals_col. O PatchCollection armazena um conjunto de patches, que no
    # caso são o conjunto de setas normais    
    normals_col = PatchCollection(normal_arrows, match_original=True, label='Normal')

    # plot dos caminhos interpolados
    # path_interp[:, 0] nesta parte roda todas as linhas da coluna 0
    # path_interp[:, 1] nesta parte roda todas as linhas da coluna 1
    ax.plot(path_interp[:, 0], path_interp[:, 1], '-', c=color, label='Interpolated')

    # adição das colunas tangentes e normais aos eixos
    ax.add_collection(tangents_col)
    ax.add_collection(normals_col)


def plot_model(img, vessel_model, cross_paths, ax):
    """ Plotando a imagem, juntamente com o modelo do vaso, com as linhas preenchidas ao longo do vaso, superior
    e inferior, na cor verde e exibição da linha medial na cor vermelha.

    Parâmetros:
    -----------
    img: ndarray, float
        imagem da área onde contém o vaso
    vessel_model: obejct VesselModel
       retorna o modelo do vaso com um objeto instanciado da classe VesselModel
    cross_paths: ndarray, float
       caminhos transversais
    ax: object, AxesSubplot
       objeto do tipo AxesSubplot que faz parte da biblioteca matplotlib
    Retorno:
    -----------
       são exibidos os caminho 1, 2 (cor verde) e linha medial (cor vermelha) interpolados, as tangentes e as normais de
       cada um destes itens
    """
    # variáveis que absorve os caminhos1 e 2 do modelo do vaso
    p1_data = vessel_model.path1
    p2_data = vessel_model.path2

    # absorvendo o caminho medial do modelo do vaso
    medial_data = vessel_model.medial_path

    # set_aspect com o parâmetro equal faz com que os eixos x e y tenham a mesma escala
    ax.set_aspect('equal')
    ax.imshow(img, 'gray')

    # chama a função que mostra os dados interpolados, as tangentes e as normais
    show_interpolated(p1_data['interpolated'], p1_data['tangents'], p1_data['normals'], ax,
                      scale=0.6, color='green')
    show_interpolated(p2_data['interpolated'], p2_data['tangents'], p2_data['normals'], ax,
                      scale=0.6, color='green')
    show_interpolated(medial_data['interpolated'], medial_data['tangents'], medial_data['normals'], ax,
                      scale=0.6, color='red')


def generate_mask(path1, path2, img_shape):
    """ Função que transforma os valores em binário

    Parâmetros:
    -----------
    path1: ndarray, float
        vetor do caminho 1
    path2: ndarray, float
        vetor do caminho 2
    img_shape: tuple, int
       informa a quantidade de linhas e colunas que a imagem do modelo do vaso conterá
    Retorno:
    -----------
    mask_img: ndarray, contendo valores True e False
       retorna a máscara para o polígono de entrada, que no caso são o path1, path2 e a img_shape
    """

    # concatenate ==> junta uma sequência de vetores ao longo das linhas
    envelop = np.concatenate((path1, path2[::-1]), axis=0)

    # round ==> arredonda uma matriz, transforma em inteiro
    envelop = np.round(envelop).astype(int)[:, ::-1]

    # tranforma a imagem em binário, passando o shape da imagem e o envelop (polígono) criado
    mask_img = draw.polygon2mask(img_shape, envelop)
    return mask_img


def create_cross_paths(cross_coord, medial_path, medial_normals, path1, path2, reach, normal_weight=2,
                       path_res_factor=3, angle_limit=45, angle_res=2):
    """ Funções relacionadas com a criação de caminhos transversais

    Parâmetros:
    -----------
    cross_coord: ndarray, float
        vetor contendo as coordenadas transversais
    medial_path: ndarray, float
        caminho medial
    medial_normals: ndarray, float
        normais do caminho medial
    path1: ndarray, float
       vetor do caminho 1
    path2: ndarray, float
       vetor do caminho 2
    reach: float
       variável que delimita o tamanho do mapa do vaso. seta o alcance superior e inferior que o
       mapa irá abranger
    normal_weight: int
        altura das normais
    path_res_factor: int
       valor que determina o quanto a resolução do caminho será aumentado. Quanto maior este valor, mais pontos serão
       criados
    angle_limit: int
        valor que determina o ângulo limite
    angle_res: int
        determina a variação que o ângulo terá
    Retorno:
    -----------
    cross_paths: list, float
        lista contendo os valores dos caminhos transversais
    cross_versors: list, float
        lista contendo os valores dos versores transversais
    """

    # criação de versores transversais
    # cria os vetores mais alinhados com as normais das linhas do envelope e da linha medial
    cross_versors = create_cross_versors(medial_path, medial_normals, path1, path2, reach, normal_weight,
                                         path_res_factor, angle_limit, angle_res)

    # transposição das coordenadas tranversais
    cross_coord = cross_coord[None].T

    cross_paths = []
    # função que pega os índices e pontos do caminho medial para criar dos caminhos tranversais
    for idxm, pointm in enumerate(medial_path):

        # pega o índice nos versores cruzados
        cross_versor = cross_versors[idxm]

        # se o versor transversal estiver vazio os caminhos transversais recebem None
        if cross_versor is None:
            cross_paths.append(None)
        else:
            # o caminho tranversal recebe o ponto + a coordenada tranversal multiplicada pelo versor transversal
            # absorve os valores e insere os pontos em uma linha transversal
            cross_path = pointm + cross_coord * cross_versor
            # os caminhos tranversais adicionam o caminho transversal em formato de lista
            cross_paths.append(cross_path.tolist())

            # retorno dos caminhos tranversais e dos versores transversais
    return cross_paths, cross_versors


def create_cross_versors(medial_path, medial_normals, path1, path2, reach, normal_weight=2,
                         path_res_factor=3, angle_limit=45, angle_res=2):
    """ Função que cria versores transversais

    Parâmetros:
    -----------
    medial_path: ndarray, float
        caminho medial
    medial_normals: ndarray, float
        normais do caminho medial
    path1: ndarray, float
       vetor do caminho 1
    path2: ndarray, float
       vetor do caminho 2
    reach: float
        variável que delimita o tamanho do mapa do vaso. seta o alcance superior e inferior que o
        mapa irá abranger
    normal_weight: int
        altura das normais
    path_res_factor: int
       valor que determina o quanto a resolução do caminho será aumentado. Quanto maior este valor, mais pontos serão
       criados
    angle_limit: int
        valor que determina o ângulo limite
    angle_res: int
        determina a variação que o ângulo terá
    Retorno:
    -----------
    cross_versors: list, float
        lista contendo os valores dos versores transversais
    """

    # definição dos ângulos -
    # concatenate ==> junta uma sequência de vetores arranjados. 
    # Os vetores tem um ângulo limite de 45 e os outros tem tamanho 2
    angles = np.concatenate((np.arange(-angle_limit, 0 + 0.5 * angle_res, angle_res),
                             np.arange(0, angle_limit + 0.5 * angle_res, angle_res)))

    # chama a função para encontrar os melhores ângulos
    idx_best_angles = find_best_angles(medial_path, medial_normals, path1, path2, angles, reach,
                                       normal_weight, path_res_factor)

    cross_versors = []

    # função que pega os índices e pontos do caminho medial para criar versores transversais
    for idxm, pointm in enumerate(medial_path):

        # pega o índice dos melhores ângulos
        idx_best_angle = idx_best_angles[idxm]

        # verificação se o índice do melhor ângulo é None
        if idx_best_angle is None:
            cross_versors.append(None)
        else:
            # criação da normal a partir
            normalm = medial_normals[idxm]
            sh_normalm = geometry.Point(normalm)
            # faz a rotação encontrando os melhores ângulos
            sh_normalm_rotated = affinity.rotate(sh_normalm, angles[idx_best_angle], origin=(0, 0))
            normalm_rotated = np.array(sh_normalm_rotated)
            cross_versors.append(normalm_rotated)
    return cross_versors


def find_best_angles(medial_path, medial_normals, path1, path2, angles, reach, normal_weight=2,
                     path_res_factor=3):
    """ Função que encontra os melhores ângulos. Faz a rotação caso seja necessário.

    Parâmetros:
    -----------
    medial_path: ndarray, float
        caminho medial
    medial_normals: ndarray, float
        normais do caminho medial
    path1: ndarray, float
       vetor do caminho 1
    path2: ndarray, float
       vetor do caminho 2
    angles: ndarray, float
        vetor que absorve os valores de limite superior e inferior dos ângulos
    reach: float
        variável que delimita o tamanho do mapa do vaso. seta o alcance superior e inferior que o
        mapa irá abranger
    normal_weight: int
        altura das normais
    path_res_factor: int
       valor que determina o quanto a resolução do caminho será aumentado. Quanto maior este valor, mais pontos serão
       criados
    Retorno:
    -----------
    idx_best_angles: list, int
        lista contendo os valores dos melhores ângulos
    """

    # os caminhos 1 e 2 são interpolados e as suas tangentes são criadas
    path1_interp, tangents1 = smutil.increase_path_resolution(path1, path_res_factor)
    path2_interp, tangents2 = smutil.increase_path_resolution(path2, path_res_factor)

    # o objeto do tipo LineString é criado passando o caminho interpolado
    sh_path1_interp = geometry.LineString(path1_interp)
    sh_path2_interp = geometry.LineString(path2_interp)

    # as normais não apontam para a mesma direção dos caminhos originais
    normals1 = smutil.get_normals(tangents1)
    normals2 = smutil.get_normals(tangents2)

    all_fitness = []
    idx_best_angles = []
    for idxm, pointm in enumerate(medial_path):
        normalm = medial_normals[idxm]
        candidate_line = np.array([pointm - reach * normalm, pointm, pointm + reach * normalm])
        sh_candidate_line = geometry.LineString(candidate_line)
        all_fitness.append([])
        for angle_idx, angle in enumerate(angles):
            sh_candidate_line_rotated = affinity.rotate(sh_candidate_line, angle)
            fitness = measure_fitness(sh_candidate_line_rotated, normalm, sh_path1_interp, normals1,
                                      sh_path2_interp, normals2, normal_weight)
            all_fitness[-1].append(fitness)

        idx_max = np.argmax(all_fitness[-1])
        if all_fitness[-1][idx_max] <= 0:
            idx_best_angles.append(None)
        else:
            idx_best_angles.append(idx_max)
            sh_candidate_line_rotated = affinity.rotate(sh_candidate_line, angles[idx_max])
            candidate_line_rotated = np.array(sh_candidate_line_rotated)
    return idx_best_angles


def measure_fitness(sh_candidate_line, normalm, sh_path1, normals1, sh_path2, normals2, normal_weight):
    """ Mede a aptidão da linha candidata.

    Parâmetros:
    -----------
    sh_candidate_line: object, LineString
        objeto do tipo shapely.geometry.linestring.LineString
    normalm: ndarray, float
        vetor contendo um par de valores
    sh_path1: object, LineString
        objeto do tipo shapely.geometry.linestring.LineString do caminho 1
    normals1: ndarray, float
       vetor contendo as normais do caminho 1
    sh_path2: object, LineString
        objeto do tipo shapely.geometry.linestring.LineString do caminho 2
    normals2: ndarray, float
       vetor contendo as normais do caminho 2
    normal_weight: int
        altura das normais
    Retorno:
    -----------
    fitness: int
        retorna se a linha candidata escolhida é a melhor opção
    """

    # tenta encontrar o ponto de intersecção dos caminhos transversais
    sh_path1_point = find_envelop_cross_path_intersection(sh_candidate_line, sh_path1)
    sh_path2_point = find_envelop_cross_path_intersection(sh_candidate_line, sh_path2)

    # se não houver intersecção a aptidão é -1, ou seja, a linha candidata não tem intersecção
    if sh_path1_point is None or sh_path2_point is None:
        fitness = -1
    else:
        path1_point = np.array(sh_path1_point)
        path2_point = np.array(sh_path2_point)
        idx_path1_point = smutil.find_point_idx(sh_path1, path1_point)
        normal1 = normals1[idx_path1_point]
        idx_path2_point = smutil.find_point_idx(sh_path2, path2_point)
        normal2 = normals2[idx_path2_point]

        candidate_line_rotated = np.array(sh_candidate_line.coords)
        candidate_normal = candidate_line_rotated[-1] - candidate_line_rotated[0]
        candidate_normal = candidate_normal / np.sqrt(candidate_normal[0] ** 2 + candidate_normal[1] ** 2)
        medial_congruence = abs(np.dot(candidate_normal, normalm))
        path1_congruence = abs(np.dot(candidate_normal, normal1))
        path2_congruence = abs(np.dot(candidate_normal, normal2))
        fitness = normal_weight * medial_congruence + path1_congruence + path2_congruence

    return fitness
