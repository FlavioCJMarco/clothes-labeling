__authors__ = ['1496622', '1494936', '1494597']
__group__ = 'DL.15'

import numpy as np
import Kmeans as km
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
import cv2
import json
from more_itertools import locate

import time
#     print("--- Tiempo de ejecución: %s segundos ---" % (time.time() - start_time))


# ANALISI QUALITATIU


def retrieval_by_color(images, labels, question, prob, llindar=0):  # question = color/s

    lista = []
    sorted_probs = []
    aux_prob = []

    for i in range(len(images)):
        for j in range(len(question)):
            if question[j] in labels[i]:
                aux_prob.append(prob[i][question[j]])
                if j == len(question) - 1:
                    lista.append(images[i])
                    sorted_probs.append(sum(aux_prob))
                    aux_prob = []
            else:
                aux_prob = []
                break

    Z = [x for (y, x) in sorted(zip(sorted_probs, lista), key=lambda pair: pair[0], reverse=True)]
    # print(sorted(sorted_probs, reverse=True))
    visualize_retrieval(Z, len(Z), sorting_element=sorted(sorted_probs, reverse=True), sorting="color")


def retrieval_by_shape(images, labels, question, max_vote_list):
    lista = []
    labels_votes = []
    for i in range(len(images)):
        if question == labels[i]:
            lista.append(images[i])
            labels_votes.append(max_vote_list[i])
    Z = [x for (y, x) in sorted(zip(labels_votes, lista), key=lambda pair: pair[0], reverse=True)]
    # print(sorted(labels_votes, reverse=True))
    visualize_retrieval(Z, len(Z), sorting_element=sorted(labels_votes, reverse=True), sorting="shape")


def retrieval_combined(images, color_labels, shape_labels, color_question, shape_question, prob, max_vote_list,
                       sorting="shape"):
    lista = []

    color_sort = False
    shape_sort = False

    if sorting == "color".lower():
        sorted_probs = []
        aux_prob = []
        color_sort = True
    elif sorting == "shape".lower():
        labels_votes = []
        shape_sort = True

    combined_labels = list(zip(color_labels, shape_labels))
    for i, labels in enumerate(combined_labels):
        bool_color = False
        bool_shape = False
        if all(color in labels[0] for color in color_question):
            bool_color = True
        if shape_question == labels[1]:
            bool_shape = True
        if bool_color and bool_shape:
            lista.append(images[i])
            if color_sort:
                for j in range(len(color_question)):
                    if color_question[j] in color_labels[i]:
                        aux_prob.append(prob[i][color_question[j]])
                        if j == len(color_question) - 1:
                            sorted_probs.append(sum(aux_prob))
                            aux_prob = []
                    else:
                        aux_prob = []
                        break
            elif shape_sort:
                labels_votes.append(max_vote_list[i])

    if color_sort:
        lista = [x for (y, x) in sorted(zip(sorted_probs, lista), key=lambda pair: pair[0], reverse=True)]
        visualize_retrieval(lista, len(lista), sorting_element=sorted(sorted_probs, reverse=True), sorting="color")
    elif shape_sort:
        lista = [x for (y, x) in sorted(zip(labels_votes, lista), key=lambda pair: pair[0], reverse=True)]
        visualize_retrieval(lista, len(lista), sorting_element=sorted(labels_votes, reverse=True), sorting="shape")
    else:
        visualize_retrieval(lista, len(lista))


#####################
#####################
# ANALISI QUANTITATIU
#####################
#####################


def kmean_statistics(element_kmeans, images, kmax=10):
    wcd = {}
    icd = {}
    fisher = {}
    iteracions = {}
    size_x = len(images[0])
    size_y = len(images[0][0])
    for i in range(2, kmax+1):
        element_kmeans.K = i
        element_kmeans.fit()
        wcd[i] = element_kmeans.withinClassDistance()
        icd[i] = element_kmeans.interclass_distance()
        fisher[i] = element_kmeans.fisher_coefficient()
        iteracions[i] = element_kmeans.num_iter

        visualize_k_means(element_kmeans, [size_x, size_y, 3])


def get_shape_accuracy(knn_labels, shape_labels):
    accuracy = 100*sum(1 for x, y in zip(sorted(knn_labels), sorted(shape_labels)) if x == y) / len(knn_labels)
    print("KNN algorithm has assigned shape labels with an accuracy of {:.2f}%".format(accuracy))


def get_color_accuracy(kmeans_labels, color_labels):
    accuracy = 0
    for i in range(len(kmeans_labels)):
        kmeans_labels_set = list(set(kmeans_labels[i]))
        for j in range(len(set(kmeans_labels_set))):
            if kmeans_labels_set[j] in color_labels[i]:
                accuracy += 1/len(color_labels[i])
    accuracy = 100*accuracy/len(color_labels)
    print("KMeans algorithm has assigned color labels with an accuracy of {:.2f}%".format(accuracy))

#####################
#####################
#####################
#####################
#####################


if __name__ == '__main__':
    start_time = time.time()

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = \
        read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    labels_complete = list(list(train_class_labels) + list(test_class_labels))

    # Kmeans and KNN elements declaration
    elements_kmeans = []
    kmeans_color_labels = []
    for i in range(len(test_imgs)):
        elements_kmeans.append(km.KMeans(test_imgs[i]))
        elements_kmeans[i].options['km_init'] = 'custom'
        elements_kmeans[i].best_find_bestK(10, "fisher", 5)
        kmeans_color_labels.append(km.get_colors(elements_kmeans[i].centroids))

    knn = KNN.KNN(train_imgs, train_class_labels)
    knn_labels = knn.predict(test_imgs, 7)
    aux_shape_labels = [None] * len(test_imgs)
    count_shape_labels = [None] * len(test_imgs)
    for i in range(len(test_imgs)):
        _, count_shape_labels[i] = np.unique(knn.neighbors[i], return_counts=True)
        count_shape_labels[i] = max(count_shape_labels[i])

    # COLOR PERCENTAGES ##############################################
    prob = {}
    prob_images = {}
    for i, label in enumerate(kmeans_color_labels):
        for j in range(len(label)):
            if label[j] not in prob:
                prob[label[j]] = 1/len(label)
            else:
                prob[label[j]] += 1/len(label)
        prob_images[i] = prob
        prob = {}

    ####################################################################################################################
    # FUNCTION CALLS ###################################################################################################
    ####################################################################################################################

    # QUALITATIVE FUNCTIONS ############################################################################################
    retrieval_by_color(test_imgs, kmeans_color_labels, ['Red', 'Blue'], prob_images)
    retrieval_by_shape(test_imgs, knn_labels, 'Flip Flops', count_shape_labels)
    retrieval_combined(test_imgs, kmeans_color_labels, knn_labels, ['Yellow', 'Black'], 'Flip Flops', prob_images,
                       count_shape_labels, sorting="color")

    # QUANTITATIVE FUNCTIONS ###########################################################################################
    np.random.seed()
    statistics_index = np.random.randint(0, len(test_imgs))
    kmean_statistics(km.KMeans(test_imgs[statistics_index]), test_imgs, 10)
    # We pass a random image given by numpy's randint function to the kmean_statistics function
    get_shape_accuracy(knn_labels, test_class_labels)
    get_color_accuracy(kmeans_color_labels, test_color_labels)

print("--- Execution time: %s seconds ---" % (time.time() - start_time))









