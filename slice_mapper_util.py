import math
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import Voronoi
from shapely import geometry, ops as shops

def arc_length(path):   
    '''Calcula o comprimento de arco acumulado, entre dois pontos'''
    
    dl = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    l = np.cumsum(dl).tolist()
    l = np.array([0] + l)
    
    return l

def interpolate(path, delta_eval=2., smoothing=0.1, k=3, return_params=False):    
    '''Interpola uma lista de pontos'''

    #transforma o path em np.array
    path = np.array(path)

    #absorve o comprimento do arco acumulado
    l = arc_length(path)

    #numero de pontos absorve o tamanho do path
    num_points = len(path)
    
    # tck características da curva
    # utilização do splprep do scipy para fazer a interpolação
    (tck, u), fp, ier, msg = splprep(path.T, s=smoothing*num_points, k=k, full_output=True)
    
    # o l na última posição é o valor acumulado de todas as somas do comprimento do arco
    delta_eval_norm = delta_eval/l[-1]
    eval_points = np.arange(0, 1+0.75*delta_eval_norm, delta_eval_norm)
    
    # pontos interpolados
    x_interp, y_interp = splev(eval_points, tck, ext=3)

    #derivadas dos pontos interpolados, der=1, tem-se a derivada e não o ponto
    dx_interp, dy_interp = splev(eval_points, tck, der=1, ext=3)
    
    #.T inverte ao invés de ter duas linhas e num_points colunas, vira num_points linhas e duas colunas
    path_interp = np.array([x_interp, y_interp]).T
    tangent = np.array([dx_interp, dy_interp]).T

    #normalizando, tamanho 1
    t_norm = np.sqrt(np.sum(tangent**2, axis=1))
    tangent = tangent/t_norm[None].T
    
    if return_params:
        return path_interp, tangent, tck, u
    else:
        return path_interp, tangent

def two_stage_interpolate(path, delta_eval=2., smoothing=0.1, k=3):
    '''Interpolar o caminho em dois estágios. Primeiro, uma interpolação linear é aplicada para que
    pontos intermediários sejam gerados. Então, uma interpolação cúbica é aplicada. Isso é útil
    porque a interpolação cúbica garante que o spline passe próximo aos pontos originais
    no caminho, mas pode estar longe da curva original entre dois pontos originais. Fazendo
    primeiro uma interpolação linear seguida por uma cúbica, o spline resultante não pode ser
    muito longe do caminho original.
    
    Parâmetros:
    -----------
    path: ndarray
        Lista de pontos contendo o caminho a ser integrado.
    delta_eval : float
        O intervalo para avaliar a interpolação.
    smoothing: float
        Fator de suavização. 0 significa que o spline passará por todos os pontos interpolados linearmente.
    k : int
        O grau da segunda interpolação - que no caso é cúbica.

    Retorno:
    -----------
    path_interp: ndarray
        caminho interpolado de forma linear e depois de forma cúbica. 
    tangent: float
        lista de tangentes
    '''
    
    path_interp_linear, _ = interpolate(path, delta_eval=delta_eval, smoothing=0, k=1)
    path_interp, tangent = interpolate(path_interp_linear, delta_eval=delta_eval, smoothing=smoothing, k=k)
 
    return path_interp, tangent

def get_normals(tangents):    
    '''Pega vetores normais mediante uma lista de vetores tangentes'''
    
    normals = np.zeros((len(tangents), 2))
    for idx, t in enumerate(tangents):
        tx, ty = t
        if ty<1e-3:
            n2 = 1
            n1 = -ty*n2/tx
        else:
            n1 = 1
            n2 = -tx*n1/ty

        norm = np.sqrt(n1**2 + n2**2)
        n = np.array([n1/norm, n2/norm])
        
        orient = np.sign(np.cross(t, n))
        if idx>0:
            if orient!=prev_orient:               
                # Se a orientação do vetor for diferente do vetor anterior, existe a inverção da orientação
                n *= -1
                orient *= -1
        prev_orient = orient
        normals[idx] = n
        
    return normals

def dist(p1, p2):
    '''Calcula a distância Euclidiana entre dois pontos'''
    
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def medial_voronoi_ridges(path1, path2):
    '''Extração das arestas mediais de Voronoi entre o caminho1 e caminho2. Diagramas de Voronoi podem ser
    usados para representar o caminho medial de uma estrutura tubular'''
    
    # cria novo array concatenado entre os caminho ao longo das linhas
    all_points = np.concatenate((path1, path2), axis=0)
    all_points_ordered = np.concatenate((path1, path2[::-1]), axis=0)

    vor = Voronoi(all_points)
    num_points_path1 = len(path1)
    tube_region = geometry.Polygon(all_points_ordered)
    idx_internal_vertices = set()    
    # Pega os vértices de Vornoi dentro do tubo
    for idx_vertex, vertex in enumerate(vor.vertices):
        if tube_region.contains(geometry.Point(vertex)):
            idx_internal_vertices.add(idx_vertex)
 
    idx_medial_vertices = []
    point_relation = []
    for idx, ridge in enumerate(vor.ridge_points):

        first_is_path1 = True if ridge[0] < num_points_path1 else False
        second_is_path1 = True if ridge[1] < num_points_path1 else False
        if (first_is_path1+second_is_path1)==1:            
            # Verificação se um cume está entre um ponto no caminho1 e outro no caminho2
            idx_ridge_vertices = vor.ridge_vertices[idx]
            if idx_ridge_vertices[0] in idx_internal_vertices and idx_ridge_vertices[1] in idx_internal_vertices:                
                # Tenha cuidado com o índice -1 em idx_ridge_vertices para pontos terminais
                idx_medial_vertices.append(idx_ridge_vertices)
                if ridge[0] < num_points_path1:
                    point_relation.append((ridge[0], ridge[1]))
                else:
                    point_relation.append((ridge[1], ridge[0]))

    idx_medial_vertices = np.array(idx_medial_vertices)
    point_relation = np.array(point_relation)
    
    return vor, idx_medial_vertices, point_relation

def order_ridge_vertices(idx_vertices):
    '''Ordena os vértices das arestas mediais de Voroni. Uma lista de arestas mediais de Voronoi, que não estão ordenados, são passados
    como parâmetro e na execução da função temos a ordenação destes vértices que definem um caminho por partes.'''
    
    idx_vertices = list(map(tuple, idx_vertices))
    vertice_ridge_map = {}
    last_vertex = -1
    for idx_ridge, (idx_v1, idx_v2) in enumerate(idx_vertices):
        if idx_v1 in vertice_ridge_map:
            vertice_ridge_map[idx_v1].append(idx_ridge)
        else:
            vertice_ridge_map[idx_v1] = [idx_ridge]

        if idx_v2 in vertice_ridge_map:
            vertice_ridge_map[idx_v2].append(idx_ridge)
        else:
            vertice_ridge_map[idx_v2] = [idx_ridge]
    
    for idx_vertex, indices_ridge in vertice_ridge_map.items():
        if len(indices_ridge)==1:
            idx_first_vertex = idx_vertex
            break
    
    ordered_vertices = [idx_first_vertex]
    idx_ridge = vertice_ridge_map[idx_first_vertex][0]
    idx_v1, idx_v2 = idx_vertices[idx_ridge]
    if idx_v1==idx_first_vertex:
        idx_vertex = idx_v2
    else:
        idx_vertex = idx_v1
    ordered_vertices.append(idx_vertex)
    prev_idx_ridge = idx_ridge
    prev_idx_vertex = idx_vertex
    while True:
        indices_ridge = vertice_ridge_map[idx_vertex]
        if len(indices_ridge)==1:
            break
        if indices_ridge[0]==prev_idx_ridge:
            idx_ridge = indices_ridge[1]
        else:
            idx_ridge = indices_ridge[0]
        idx_v1, idx_v2 = idx_vertices[idx_ridge]
        if idx_v1==prev_idx_vertex:
            idx_vertex = idx_v2
        else:
            idx_vertex = idx_v1
        
        ordered_vertices.append(idx_vertex)
        prev_idx_ridge = idx_ridge
        prev_idx_vertex = idx_vertex
    
    return ordered_vertices
        
def invert_if_oposite(path1, path2):
    '''Inverte o caminho2, se o caminho1 e o caminho2 forem demarcados em direções opostas. 
    Isto acontece quando demarcamos os vasos um da direita para a esquerda e outro da esquerda para a direita, ou vice-versa.'''
    
    min_size = min([len(path1), len(path2)])
    avg_dist = np.sum(np.sqrt(np.sum((path1[:min_size]-path2[:min_size])**2, axis=1)))
    avg_dist_inv = np.sum(np.sqrt(np.sum((path1[:min_size]-path2[::-1][:min_size])**2, axis=1)))
    if avg_dist_inv<avg_dist:
        # Inverção do vetor de caminhos2
        path2 = path2[::-1]
    
    return path2

def increase_path_resolution(path, res_factor):

    '''Incrementa a resolução de um dado caminho, através da aplicação de um fator'''
    
    x, y = path.T
    num_points = len(path)
    indices = list(range(num_points))
    # Define a variável paramétrica certificando-se de que ela passe por todo o ponto original
    tck, _ = splprep(path.T, u=indices, s=0, k=3)
    eval_points = np.linspace(0, num_points-1, num_points*res_factor - (res_factor-1))
    x_interp, y_interp = splev(eval_points, tck, der=0)
    x_tangents, y_tangents = splev(eval_points, tck, der=1)
    
    path_interp = np.array([x_interp, y_interp]).T
    tangents = np.array([x_tangents, y_tangents]).T
    
    return path_interp, tangents

def find_point_idx(sh_path, point):
    '''Encontra índice do point em sh_path'''

    #aplicação da distância Euclidiana para encontrar as menores distâncias entre dois pontos
    dists = np.sqrt((sh_path.xy[0]-point[0])**2+(sh_path.xy[1]-point[1])**2)

    #retorna as distâncias mínimas
    return np.argmin(dists)
    