import numpy as np
from matplotlib.patches import Arrow, FancyArrowPatch, FancyArrowPatch, ArrowStyle, FancyArrow
from matplotlib.collections import PatchCollection, PathCollection
from scipy.ndimage import map_coordinates
import dash_components.slice_mapper_util as smutil
from shapely import geometry, ops as shops, affinity
from IPython.display import display
import dash_components.slice_mapper_util as sm_util
from skimage import draw


class SliceMapper:

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
    '''Gera um envelope suave ao longo do caminho1 e caminho2
    o smutil é dash_components.slice_mapper_util
    '''
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
   
    ''' Extrai o caminho medial a partir de uma estrutura tubular'''

    vor, idx_medial_vertices, point_relation = smutil.medial_voronoi_ridges(path1_interp, path2_interp)
    idx_medial_vertices_ordered = smutil.order_ridge_vertices(idx_medial_vertices)
    medial_path = []
    for idx_vertex in idx_medial_vertices_ordered:
        medial_path.append(vor.vertices[idx_vertex])
    medial_path = np.array(medial_path)
    medial_path = smutil.invert_if_oposite(path1_interp, medial_path)

    # Garante que o caminho medial vai até o final do tubo
    first_point = (path1_interp[0] + path2_interp[0]) / 2
    last_point = (path1_interp[-1] + path2_interp[-1]) / 2
    medial_path = np.array([first_point.tolist()] + medial_path.tolist() + [last_point.tolist()])
    medial_path_info = interpolate_medial_path(medial_path, delta_eval=delta_eval, smoothing=smoothing)

    if return_voronoi:
        return medial_path, medial_path_info, vor
    else:
        return medial_path, medial_path_info


def create_cross_paths_old(path, normals, cross_coord, remove_endpoints=True, return_flat=True):
    if remove_endpoints:
        # É útil remover endpoints se o caminho foi interpolado
        path = path[1:-1]
        normals = normals[1:-1]

    cross_coord = cross_coord[None].T
    cross_paths = []
    for point_idx, point in enumerate(path):
        cross_path = point + cross_coord * normals[point_idx]
        cross_paths.append(cross_path.tolist())
    if return_flat:
        cross_paths = [point for cross_path in cross_paths for point in cross_path]
    cross_paths = np.array(cross_paths)

    return cross_paths


def create_cross_paths_limit(path, normals, cross_coord, remove_endpoints=True):
    ''' Esta função cria os limites das trajetórias transversais'''

    if remove_endpoints:
        # É útil remover endpoints (pontos de parada - É ISSO MESMO?) se o caminho foi interpolado
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

    ''' Criação do modelo do vaso'''

    #chama a função de inversão. Se o caminho estiver invertido o caminho2 é invertido
    path2 = smutil.invert_if_oposite(path1, path2)

    # variáveis absorvem o resultado do envelopamento de caminho1, caminho2, passamos um delta_eval 
    # que aumenta a resolução e um grau de suavização é aplicado
    path1, path1_info, path2, path2_info = interpolate_envelop(path1, path2, delta_eval, smoothing)

    # As informação contidas nos caminhos 1 e 2 são inseridas nas variáveis
    path1_interp, tangents1, normals1 = path1_info
    path2_interp, tangents2, normals2 = path2_info

    # A linha medial, juntamente com suas informações são criadas
    medial_path, medial_path_info = extract_medial_path(path1_interp, path2_interp, delta_eval=delta_eval,
                                                        smoothing=smoothing)

    # o modelo vaso é criado e passado como retorno da função
    vm = VesselModel(path1, path1_info, path2, path2_info, medial_path, medial_path_info, delta_eval)

    return vm


def create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=False):
    
    '''Cria uma imagem contendo intensidades de seção transversal ao longo do caminho medial fornecido'''

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
    mask_img = generate_mask(path1_interp, path2_interp, img.shape)
    mapped_mask_values = map_coordinates(mask_img, cross_paths_flat.T[::-1], output=np.uint8,
                                         order=0, mode='mirror')
    mapped_mask_values = mapped_mask_values.reshape(-1, len(cross_coord)).T

    # pega as precisas posições para o caminho1 e caminho2 interpolado no mapa 
    path1_mapped, path2_mapped = find_vessel_bounds_in_map(path1_interp,
                                                           path2_interp, cross_paths_valid, delta_eval, smoothing)

    vessel_map = VesselMap(mapped_values, medial_coord, cross_coord, cross_versors, mapped_mask_values, path1_mapped,
                           path2_mapped)

    if return_cross_paths:
        return vessel_map, cross_paths_valid
    else:
        return vessel_map


def find_vessel_bounds_in_map(path1_interp, path2_interp, cross_paths, delta_eval, smoothing):

    '''Encontra os limites dos vasos no mapa'''

    # LineString: O objeto LineString construído representa um ou mais splines lineares conectados entre os pontos. 
    # Pontos repetidos na sequência ordenada são permitidos, mas podem incorrer em penalidades de desempenho e 
    # devem ser evitados. Uma LineString pode se cruzar ( ou seja , ser complexa e não simples ).
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

    path1_mapped = np.array(path1_mapped) / delta_eval
    path2_mapped = np.array(path2_mapped) / delta_eval

    #retorno dos valores do caminho 1 e 2 mapeados
    return path1_mapped, path2_mapped


def find_envelop_cross_path_intersection(sh_cross_path, sh_path_interp, max_dist_factor=2.):

    '''Encontra interseções dos caminhos cruzados do envelope'''

    #pega o índice inteiro do meio do tamanho de sh_cross_path.coords
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

    ''' Criando as fatias do mapa'''
    
    # criação do modelo do vaso
    vessel_model = create_vessel_model(img, path1, path2, delta_eval, smoothing)

    # criação do mapa do vaso e dos caminhos transversais
    vessel_map, cross_paths = create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=True)
    vessel_model.set_map(vessel_map)

    # retornando o modelo do vaso e os caminhos transversais
    return vessel_model, cross_paths


def interpolate_medial_path(path, delta_eval=2., smoothing=0.01):
    
    '''Interpolando o caminho medial'''

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
    """Show interpolated path, together with tangents and normals. scale defines the length
    of the arrows."""

    tangent_heads = path_interp + scale * tangents
    normals_heads = path_interp + scale * normals
    arrow_style = ArrowStyle("->", head_length=10, head_width=3)
    tangent_arrows = []
    for idx in range(len(path_interp)):
        # fa = FancyArrowPatch(path_interp[idx], tangent_heads[idx], arrowstyle=arrow_style, color='orange')
        fa = FancyArrow(path_interp[idx, 0], path_interp[idx, 1], scale * tangents[idx, 0], scale * tangents[idx, 1],
                        width=0.01, head_width=0.1, head_length=0.2, color='orange')
        tangent_arrows.append(fa)
    tangents_col = PatchCollection(tangent_arrows, match_original=True, label='Tangent')

    normal_arrows = []
    for idx in range(len(path_interp)):
        # fa = FancyArrowPatch(path_interp[idx], normals_heads[idx], arrowstyle=arrow_style, color='orange')
        fa = FancyArrow(path_interp[idx, 0], path_interp[idx, 1], scale * normals[idx, 0], scale * normals[idx, 1],
                        width=0.01, head_width=0.1, head_length=0.2, color='orange')
        normal_arrows.append(fa)
    normals_col = PatchCollection(normal_arrows, match_original=True, label='Normal')

    ax.plot(path_interp[:, 0], path_interp[:, 1], '-', c=color, label='Interpolated')
    ax.add_collection(tangents_col)
    ax.add_collection(normals_col)
    # for ta, na in zip(tangent_arrows, normal_arrows):
    #    ax.add_patch(ta)
    #    ax.add_patch(na)


def plot_model(img, vessel_model, cross_paths, ax):
    p1_data = vessel_model.path1
    p2_data = vessel_model.path2
    medial_data = vessel_model.medial_path

    x1, y1 = p1_data['original'].T
    x2, y2 = p2_data['original'].T
    ax.set_aspect('equal')
    ax.imshow(img, 'gray')
    # ax.plot(x1, y1, '-o', c='blue', label='Original1')
    # ax.plot(x2, y2, '-o', c='blue', label='Original2')
    show_interpolated(p1_data['interpolated'], p1_data['tangents'], p1_data['normals'], ax,
                      scale=0.6, color='green')
    show_interpolated(p2_data['interpolated'], p2_data['tangents'], p2_data['normals'], ax,
                      scale=0.6, color='green')
    show_interpolated(medial_data['interpolated'], medial_data['tangents'], medial_data['normals'], ax,
                      scale=0.6, color='red')
    # for cross_path in cross_paths:
    ##p1, p2 = cross_paths[idx_path], cross_paths[idx_path+cross_coord.shape[0]-1]
    ##plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o', c='cyan')
    ##cross_path = cross_paths[idx_path:idx_path+cross_coord.shape[0]]
    # ax.plot(cross_path[:,0], cross_path[:,1], '-o', c='cyan', ms=3, alpha=0.2)

    # ax.legend(loc=2)


def generate_mask(path1, path2, img_shape):
    envelop = np.concatenate((path1, path2[::-1]), axis=0)
    envelop = np.round(envelop).astype(int)[:, ::-1]
    mask_img = draw.polygon2mask(img_shape, envelop)

    return mask_img


# Functions related to the creation of cross-sectional paths
def create_cross_paths(cross_coord, medial_path, medial_normals, path1, path2, reach, normal_weight=2,
                       path_res_factor=3, angle_limit=45, angle_res=2):
    cross_versors = create_cross_versors(medial_path, medial_normals, path1, path2, reach, normal_weight,
                                         path_res_factor, angle_limit, angle_res)
    cross_coord = cross_coord[None].T
    cross_paths = []
    for idxm, pointm in enumerate(medial_path):
        cross_versor = cross_versors[idxm]
        if cross_versor is None:
            cross_paths.append(None)
        else:
            cross_path = pointm + cross_coord * cross_versor
            cross_paths.append(cross_path.tolist())

            # plt.plot([pointm[0]-3*normalm[0], pointm[0], pointm[0]+3*normalm[0]],
            #         [-pointm[1]+3*normalm[1], -pointm[1], -pointm[1]-3*normalm[1]], c='blue')
            # plt.plot([pointm[0]-3*normalm_rotated[0], pointm[0], pointm[0]+3*normalm_rotated[0]],
            #         [-pointm[1]+3*normalm_rotated[1], -pointm[1], -pointm[1]-3*normalm_rotated[1]], c='red')

    return cross_paths, cross_versors


def create_cross_versors(medial_path, medial_normals, path1, path2, reach, normal_weight=2,
                         path_res_factor=3, angle_limit=45, angle_res=2):
    angles = np.concatenate((np.arange(-angle_limit, 0 + 0.5 * angle_res, angle_res),
                             np.arange(0, angle_limit + 0.5 * angle_res, angle_res)))
    idx_best_angles = find_best_angles(medial_path, medial_normals, path1, path2, angles, reach,
                                       normal_weight, path_res_factor)

    cross_versors = []
    for idxm, pointm in enumerate(medial_path):
        idx_best_angle = idx_best_angles[idxm]
        if idx_best_angle is None:
            cross_versors.append(None)
        else:
            normalm = medial_normals[idxm]
            sh_normalm = geometry.Point(normalm)
            sh_normalm_rotated = affinity.rotate(sh_normalm, angles[idx_best_angle], origin=(0, 0))
            normalm_rotated = np.array(sh_normalm_rotated)
            cross_versors.append(normalm_rotated)

    return cross_versors


def find_best_angles(medial_path, medial_normals, path1, path2, angles, reach, normal_weight=2,
                     path_res_factor=3):
    path1_interp, tangents1 = smutil.increase_path_resolution(path1, path_res_factor)
    path2_interp, tangents2 = smutil.increase_path_resolution(path2, path_res_factor)
    sh_path1_interp = geometry.LineString(path1_interp)
    sh_path2_interp = geometry.LineString(path2_interp)
    # Warning, normals do not point to the same direction as in the original paths
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
            # plt.plot([candidate_line_rotated[0][0], candidate_line_rotated[-1][0]], [-candidate_line_rotated[0][1], -candidate_line_rotated[-1][1]])
            # plt.plot([pointm[0], pointm[0]+normalm[0]], [-pointm[1], -pointm[1]-normalm[1]], c='blue')

    # import pdb; pdb.set_trace()

    return idx_best_angles


def measure_fitness(sh_candidate_line, normalm, sh_path1, normals1, sh_path2, normals2, normal_weight):
    sh_path1_point = find_envelop_cross_path_intersection(sh_candidate_line,
                                                          sh_path1)
    sh_path2_point = find_envelop_cross_path_intersection(sh_candidate_line,
                                                          sh_path2)
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

        # plt.plot([candidate_line_rotated[0][0], candidate_line_rotated[-1][0]], [-candidate_line_rotated[0][1], -candidate_line_rotated[-1][1]])
        # plt.plot([path1_point[0], path2_point[0]], [-path1_point[1], -path2_point[1]], 'o', c='blue')
        # plt.plot([path1_point[0], path1_point[0]+normal1[0]], [-path1_point[1], -path1_point[1]-normal1[1]], c='orange')
        # plt.plot([path2_point[0], path2_point[0]+normal2[0]], [-path2_point[1], -path2_point[1]-normal2[1]], c='orange')

    return fitness
