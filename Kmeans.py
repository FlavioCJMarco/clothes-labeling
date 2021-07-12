__authors__ = ['1496622', '1494936', '1494597']
__group__ = 'DL.15'

import numpy as np
from scipy.spatial.distance import cdist
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        # self.centroids = 0
        # self.X = 0
        self._init_X(X)
        self._init_options(options)  # DICT options

    ##############################################################
    #  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed  #
    ##############################################################
        # self.labels = 0
        # self.old_centroids = 0

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        if isinstance(X, float) is False:
            # Si X NO es float la convertimos en float y almacenamos el valor en self.X
            self.X = X.astype(np.float)
        if len(X[0]) != 3:
            # Si X no tiene 3 columnas (para r, g y b) se calcula el número de filas (val) y se
            # cambia el tamaño de la matriz X por el de val x 3
            val = np.int64(np.divide(np.size(self.X), 3))
            self.X = np.reshape(self.X, (val, 3))

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0.00315
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        ###########################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION  #
        #  AND CHANGE FOR YOUR OWN CODE                           #
        ###########################################################

        first = False
        rand = False
        custom = False

        self.old_centroids = np.empty([self.K, 3])
        self.centroids = np.empty([self.K, 3], np.float64)
        self.centroids[:] = np.nan

        if self.options['km_init'].lower() == 'custom':
            custom = True
            # Inicializamos la variable rand_index como un número al azar del número de filas:
            rows = len(self.X)
            rand_index = np.random.choice(rows)
            self.centroids[0] = self.X[rand_index]

            # Calculamos la distancia desde el primer centroide al resto de píxeles.
            distances = cdist(self.X, np.array([self.centroids[0]])).flatten()

        elif self.options['km_init'].lower() == 'first':
            first = True
            i = 0
        elif self.options['km_init'].lower() == 'random':
            rand = True
            i = 0
            np.random.seed()
            aux = np.random.randint(low=0, high=len(self.X)-1)

        if custom:
            for i in range(1, self.K):
                # Para elegir el próximo centroide, la probabilidad de que cada píxel lo sea
                # es directamente proporcional a la distancia con el centroide más cercano al
                # al cuadrado
                prob = distances ** 2
                rand_index = np.random.choice(rows, size=1, p=prob / np.sum(prob))
                self.centroids[i] = self.X[rand_index]

                if i == self.K - 1:
                    break
                # Si necesitamos otro centroide, calculamos de nuevo las distancias desde el nuevo centroide
                # a todos los píxeles y establecemos la distancia a cada uno como la mínima entre
                # la que tenían con el centroide anterior y la que tienen con el nuevo centroide
                distances_new = cdist(self.X, np.array([self.centroids[i]]), metric='euclidean').flatten()
                distances = np.min(np.vstack((distances, distances_new)), axis=0)

        if first:
            for pixel in self.X:
                if not any((np.equal(pixel, self.centroids).all(1))):
                    # np.equal(...).all(1) te devuelve en pequeñas listas de 3 booleanos.
                    # Para cada lista que NO sea (True, True, True) => any(...) devolverá
                    # False, que es lo que buscamos, por eso hacemos not any(...).
                    # Si la lista es (True, True, True) significa que el centroide ya existía.
                    self.centroids[i] = pixel
                    i = np.add(i, 1)
                    if i == self.K:
                        break
        elif rand:
            while i != self.K:
                if not any((np.equal(self.X[aux], self.centroids).all(1))):
                    self.centroids[i] = self.X[aux]
                    i = np.add(i, 1)
                np.random.seed()
                aux = np.random.randint(low=0, high=len(self.X) - 1)


    def get_labels(self):
        """Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """

        # self.labels = np.empty(len(self.X), dtype=np.int64)
        # self.labels[:] = np.nan
        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = np.array(self.centroids, copy=True)
        # suma_rgb = np.empty((self.K, 3), dtype=np.float64)
        # pixels_per_centroid = np.bincount(self.labels)
        # pixels_per_centroid = pixels_per_centroid.reshape(-1, 1)
        # suma_rgb = [(self.X[self.labels == k].sum(0)) for k in range(self.K)]
        self.centroids = np.divide([(self.X[self.labels == k].sum(0)) for k in range(self.K)], np.bincount(
            self.labels).reshape(-1, 1))

        """ 
        for i in range(0, len(self.centroids)):
            pixels_per_centroid[i] = np.where(self.labels == i)[0]

            # self.X[p] contiene los valores de red, green y blue, a los que accedemos con el siguiente índice
            r = [self.X[p][0] for p in pixels_per_centroid[i]]  # CAMBIAR ESTOS TRES FORS POR UNO SOLO
            g = [self.X[p][1] for p in pixels_per_centroid[i]]
            b = [self.X[p][2] for p in pixels_per_centroid[i]]

            self.centroids[i] = np.array([sum(r) / len(pixels_per_centroid[i]), sum(g) / len(pixels_per_centroid[i]),
                                          sum(b)/len(pixels_per_centroid[i])])
        """

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        # OJO: El test nunca retorna False, pero multiplicando por cualquier valor alguna de las arrays
        # comprobamos que el return sí puede ser False.
        return np.allclose(self.old_centroids, self.centroids, rtol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter = np.add(self.num_iter, 1)

    def withinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        # distancias = distance(self.X, self.centroids)
        # dist_cluster = np.zeros(len(self.X))

        # for pixel in range(0, len(distancias)):
        #   dist_cluster[pixel] = np.amin(distancias)[pixel]

        # dist_cluster = np.amin(distance(self.X, self.centroids), axis=1)

        self.wcd = np.multiply(np.divide(1, len(self.X)), np.sum(np.square(np.amin(distance(self.X, self.centroids), axis=1))))
        return self.wcd
        # return wcd


    def interclass_distance(self):
        self.icd = np.mean(distance(np.array(self.centroids), self.centroids)[np.nonzero(distance(np.array(
            self.centroids), self.centroids))])
        return self.icd

    def fisher_coefficient(self):
        self.fisher = self.withinClassDistance()/self.interclass_distance()
        return self.fisher


    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        self.K = 2
        self.fit()
        wcd = self.withinClassDistance()

        for k in range(3, max_K):
            self.K = k
            self.fit()

            # print(wcd, k)
            wcd_old = wcd
            wcd = self.withinClassDistance()
            # dec = 100*wcd/wcd_old  # Porcentaje de decremento del wcd

            if np.subtract(1, np.divide(wcd, wcd_old)) < 0.20:
                self.K = np.subtract(self.K, 1)
                self.fit()
                break

    def best_find_bestK(self, max_K, heuristics, llindar=10):
        self.K = 2
        self.fit()
        if heuristics == "icd":
            heur_element = self.interclass_distance()
        elif heuristics == "fisher":
            heur_element = self.fisher_coefficient()
        else:  # wcd = default
            heur_element = self.withinClassDistance()

        for k in range(3, max_K):
            self.K = k
            self.fit()

            # print(wcd, k)
            heur_element_old = heur_element
            if heuristics == "icd":
                heur_element = self.interclass_distance()
            elif heuristics == "fisher":
                heur_element = self.fisher_coefficient()
            else:
                heur_element = self.withinClassDistance()
            # dec = 100*wcd/wcd_old  # Porcentaje de decremento del wcd

            if np.subtract(1, np.divide(heur_element, heur_element_old)) < llindar/100:
                self.K = np.subtract(self.K, 1)
                self.fit()
                break

        # print("La millor k obtinguda amb {} ha sigut {}".format(heuristics, self.K))


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    # dist = np.sqrt(((X[:, :, None] - C[:, :, None].T) ** 2).sum(1))
    return np.sqrt(np.add(np.add(np.square((np.subtract(X[:, 0, np.newaxis], C[:, 0]))),
                                 np.square(np.subtract(X[:, 1, np.newaxis], C[:, 1]))),
                          np.square(np.subtract(X[:, 2, np.newaxis], C[:, 2]))))


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    color_probs = utils.get_color_prob(centroids)
    labels = np.empty(len(centroids), dtype=object)
    # labels[:] = np.nan

    for c in range(len(centroids)):
        # index_prob = np.argmax(color_probs[c])
        labels[c] = utils.colors[np.argmax(color_probs[c])]
    return labels
