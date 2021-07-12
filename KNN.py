__authors__ = ['1496622', '1494936', '1494597']
__group__ = 'DL.15'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if isinstance(train_data, float) is False:
            # Si train_data NO es float la convertimos en float y almacenamos el valor en self.train_data
            self.train_data = train_data.astype(np.float)

        # Tenemos:
        # IMAGEN 1: PIXEL1, PIXEL2, P3, P4, ... P14400
        # IMAGEN 2: PIXEL1, PIXEL2, P3, P4, ... P4800
        # Y así.
        self.train_data = np.reshape(self.train_data, (len(train_data), np.int64(np.divide(np.size(self.train_data),
                                                                                           len(train_data)))))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
                 Lo de arriba significa:
                 Cada fila de self.neighbors es la columna más cercana de train_data a cada fila de test_data
        """
        test_data = test_data.reshape(len(test_data), test_data[0].size)

        if isinstance(test_data, float) is False:
            test_data = test_data.astype(np.float)

        self.neighbors = self.labels[np.argsort(cdist(test_data, self.train_data))[:, :k]]

        # ¿Cómo era antes esa última línea incomprensible? ¡Así!:
        """
        distancias = cdist(test_data, self.train_data)

        i = 0

        for prenda in distancias:
            self.neighbors[i] = self.labels[np.argsort(prenda)[:k]]
            i += 1
        """


        # 1- Calcular la matriz de distancias entre los datos de test y los de entrenamiento con cdist
        # Si los conjuntos de datos son m1 x n y m2 x n, la matriz de distancias es de n x n
        #
        # NOTA: Ahora esta función está dentro del argsort, al principio. Se accede a las 0:len(test_data) filas
        # y las 0:k columnas de la matriz. Anteriormente (con el for) se hacía a las 0:k posiciones de cada fila.

        # 2- Recorremos la matriz de distancias línea a línea asignando a self.neighbors[línea] los k índices
        # correspondiente a las columnas que contengan los k elementos más pequeños, estando ordenados de menor
        # a mayor valor contenido. Este valor representa la "distancia" a un label, por lo que a partir de  los índices
        # de los k menores valores conseguimos conocer los k labels que más posiblemente se correspondan con la imagen.

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """

        output = {}  # Diccionario donde almacenaremos los valores de cada fila de self.neighbors() y su conteo
        lista = np.empty([1, 0], dtype=object)  # Array a la que iremos añadiendo el ganador de cada votación
        for counter, row in enumerate(self.neighbors):
            # Recorremos cada fila de self.neighbors
            for prenda in row:
                # Recorremos cada una de las prendas de cada fila
                if prenda not in output:
                    # Si una prenda no es key del diccionario, se le asigna un 1 como value (1 coincidencia)
                    output[prenda] = 1
                else:
                    # Si una prenda ya estaba en el diccionario, se le suma 1 a su value (1 coincidencia más)
                    output[prenda] = np.add(output[prenda], 1)
            lista = np.insert(lista, counter, (max(output, key=output.get)))
            # Insertamos el elemento en la posición del más votado (counter)
            output = {}
            # Reseteamos el diccionario
        return lista

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()


