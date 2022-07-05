#import all needed packages
from skimage import exposure
from skimage.feature import hog
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import cv2
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# for seaborn : pip install seaborn==0.11.1

np.random.seed(0)

# Neta (Windows)
data_path = r'C:\Users\nbargil\Downloads\101_ObjectCategories'
# # Ricky (Linux)
# data_path = "/home/ricky/Desktop/CV__Programming_Task_1/101_ObjectCategories"

fold_1 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Tuning
fold_2 = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Test


def GetDefaultParameters():
    '''
    This function returns all the parameters initialized for current experiment
    :return: dictionary of sub dictionaries parameters needed for main function
    Data: dictionary needed for GetData function
        Path: path for the Caletch101 main folder
        S: side length parameter to resize images
        class_indices: list of chosen classes indices for train and test
    Split: split size needed for TrainTestSplit function. train UCL number of images per class
    Prepare: dictionary needed for Prepare function
        PixelsPerCell: PixelsPerCell needed for HOG representation
        NumberOfOrientationBins: NumberOfOrientationBins needed for HOG representation
    Train: dictionary needed for TrainWithTuning and Train functions
        TrainValidationRatio: the ratio use to split train data set into train and validation for Hyper parameter tuning
        Kernel: the final Kernel chosen for SVM
        C: the final C value chosen for SVM
    '''
    parameters = {'Data': {'Path': data_path, 'S': 256, 'class_indices': fold_2},
                  'Split': 25,
                  'Prepare': {'PixelsPerCell': 16, 'NumberOfOrientationBins': 8},
                  'Train': {'TrainValidationRatio': 0.75, 'Kernel': 'linear', 'C': 100}}
    return parameters


def GetData(param_dict):
    '''
    This function loads the images from the classes indices: images for classification, corresponding images labels and info regarding all chosen classes.
    :param param_dict: Data dictionary of parameters. GetDefaultParameters output.
    :return: dictionary containing data needed for experiment
    Data: 3D array of all images from chosen classes with N X S X S dimensions (N number images, S side length parameter of image size)
    Labels: array of ints of all images labels with 1 x N dimensions
    ClassesInfo: dictionary per class containing class number as key, class name, number of images total as values
    '''

    param_data_path = param_dict['Path']
    param_data_S = param_dict['S']
    class_indices = param_dict['class_indices']
    class_folders = sorted(os.listdir(param_data_path))  # sorting 101_ObjectCategories folder alphabetically
    images_array = []
    labels_vector = []
    classes_info_dict = {}
    for indices in class_indices:
        class_folder = class_folders[indices]
        folder_path = param_data_path + '\\' + class_folder
        # folder_path = param_data_path + '/' + class_folder
        images = sorted(os.listdir(folder_path))  # sorting image files in the current folder
        classes_info_dict[indices + 1] = {'Name': class_folder, 'Images': len(images)}
        for img in images:
            img_path = folder_path + '\\' + img
            # img_path = folder_path + '/' + img
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # load image in gry scale
            img = cv2.resize(img, (param_data_S, param_data_S)) # resize image into S size
            images_array.append(np.asarray(img))
            labels_vector.append(indices + 1)

    images_array = np.array(images_array)
    labels_vector = np.array(labels_vector)
    return {'Data': images_array, 'Labels': labels_vector, 'ClassesInfo': classes_info_dict}


def TrainTestSplit(data_set, labels_vector, max_size, class_info_dict):
    '''
    This function splits the data of images and labels into train set and test set, and returns the split subsets.
    We will train our pipe on 25 images per class, test on 25 others.
    If there are less than 50 images for the class, we will train on half the images (number rounded up if the number of images is odd),
    and test on the other half.
    :param data_set: 3D array of all images from chosen classes. GetData function output.
    :param labels_vector: int list of all images labels. GetData function output.
    :param max_size: the max size of images per class. GetDefaultParameters function output, Split value.
    :param class_info_dict: dictionary info per class to update. GetData function output.
    :return: dictionary containing the images and labels split subsets of train and test datasets.
    TrainData: 3D array of images that will be used for training from chosen classes with N X S X S dimensions
    TestData: 3D array of images that will be used for testing from chosen classes with N X S X S dimensions
    TrainLabels: array of ints of all the train set images labels with N x 1 dimensions
    TestLabels: array of ints of all the test set images labels with N x 1 dimensions
    TestIndicesTuples: list of tupels containing for each image in test set: original index in Data array and index in TestData array post split
    '''

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    test_indices_tuples = []
    labels_set = class_info_dict.keys()
    index = 0
    for label in labels_set:
        number_of_objects = class_info_dict[label]['Images']
        split_size = min(max_size, math.ceil(number_of_objects / 2))

        class_info_dict[label]['Number of images used for training'] = split_size # update class info dictionary with number of images used for training
        class_info_dict[label]['Number of images used for testing'] = min(number_of_objects - split_size, max_size)  # update class info dictionary with of images used for testing
        for i in range(split_size):
            train_labels.append(np.asarray(labels_vector[index + i]))
            train_data.append(np.asarray(data_set[index + i]))
            if (i == (split_size - 1)) and (split_size != min(number_of_objects - split_size, max_size)):
                break

            else:
                test_image_original_index = i + split_size
                test_image_new_index = len(test_data)
                test_indices_tuples.append((test_image_original_index, test_image_new_index))
                test_data.append(np.asarray(data_set[index + i + split_size]))
                test_labels.append(np.asarray(labels_vector[index + i + split_size]))

        index += number_of_objects

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return {'TrainData': train_data, 'TestData': test_data,
            'TrainLabels': train_labels, 'TestLabels': test_labels,
            'TestIndicesTuples': test_indices_tuples}


def Prepare(data_set, param_dict):
    '''
    This function creates the images representation for SVM using HOG
    :param data_set: data set of images to convert to HOG. TrainTestSplit function output
    :param param_dict: Prepare dictionary of parameters. GetDefaultParameters output.
    :return: array of HOG representation vectors per image, with number of images in data_set X 7200 dimensions (7200 = 8(orientations) X 4(cells_per_block 2,2) X 15^2(number of blocks in image)
    '''
    hog_matrix = []
    pixels_per_cell = param_dict['PixelsPerCell']
    number_of_orientation_bins = param_dict['NumberOfOrientationBins']
    for image in range(data_set.shape[0]):
        hog_vector = hog(data_set[image], orientations=number_of_orientation_bins,
                         pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                         cells_per_block=(2, 2), feature_vector=True)  # ,visualize=True)
        hog_matrix.append(np.asarray(hog_vector))
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 20))
        # cv2.imshow('image', hog_image_rescaled)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    hog_matrix = np.array(hog_matrix)
    return hog_matrix


def printTunePlot(param, parmeters_values, errors, optimal_value, min_error):
    '''
    This function prints the tuning plots on screen
    :param param: string of name of parameter tuned
    :param parmeters_values: list of parameter value to tune
    :param errors: list of MSE per parameter value
    :param optimal_value: the value of parameter with minimal MSE
    :param min_error: minimal MSE obtained for optimal_value
    :return: plot showing MSE (Y axis) as function of parameter value (X axis) on current tuned parameter
    '''

    plt.subplot(1, 1, 1)
    plt.semilogx(parmeters_values, errors, label='Validation')
    plt.vlines(optimal_value, 0, min_error, color='k',
               linewidth=3, label='Optimum on Validation')
    plt.legend(loc='lower left')
    plt.ylim([0, 1])
    plt.xlabel('Parameter Value')
    plt.ylabel('Performance [MSE]')
    fig = plt.gcf()
    fig.canvas.set_window_title(param)
    plt.title(param)
    return plt


def TuneC(model, data_set_rep, train_labels_vector, key):
    '''
    This function preform the tuning for Hyper parameter C for SVM model and show the tune plot on screen
    :param model: model chosen to tune C. SVM model with unique kernel
    :param data_set_rep: HOG representation of training set images. Prepare function output.
    :param train_labels_vector: labels array for training set images. TrainTestSplit function output.
    :param key: string of current kernel
    :return: float of the Optimal C for current model tuned
    '''
    c_vec = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    cv_scores = []
    for Ci in c_vec:
        model.set_params(C=Ci)
        scores = cross_val_score(model, data_set_rep, train_labels_vector,
                                 cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]
    best_MSE = min(MSE)

    optimal_c_index = MSE.index(best_MSE)
    optimal_c = c_vec[optimal_c_index]

    plt = printTunePlot('C' + ' (' + str(key) + ')', c_vec, MSE, optimal_c, best_MSE)
    plt.show()

    return optimal_c


def TuneGammaOnRBFModel(data_set_rep, train_labels_vector):
    '''
    This function preform the tuning for Hyper parameter gamma for SVM model with RBF kernel and show the tune plot on screen
    :param data_set_rep: HOG representation of training set. Prepare function output.
    :param train_labels_vector: labels array for training set images. TrainTestSplit function output.
    :return: float of the Optimal gamma for SVM model with RBF kernel
    '''
    gamma_vec = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    cv_scores = []
    for gamma in gamma_vec:
        rsvm_clf = SVC(kernel='rbf', gamma=gamma)
        scores = cross_val_score(rsvm_clf, data_set_rep, train_labels_vector,
                                 cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]
    best_MSE = min(MSE)

    optimal_gamma_index = MSE.index(best_MSE)
    optimal_gamma = gamma_vec[optimal_gamma_index]

    plt = printTunePlot('gamma', gamma_vec, MSE, optimal_gamma, best_MSE)
    plt.show()

    return optimal_gamma


def TuneDegreeOnPolyModel(data_set_rep, train_labels_vector):
    '''
    This function preform the tuning for Hyper parameter polynomial degree for SVM model with polynomial kernel and show the tune plot on screen
    :param data_set_rep: HOG representation of training set images. Prepare function output.
    :param train_labels_vector: labels array for training set images. TrainTestSplit function output.
    :return: int of the Optimal polynomial degree for SVM model with polynomial kernel
    '''
    degree_vec = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    cv_scores = []
    for degree in degree_vec:
        psvm_clf = SVC(kernel='poly', degree=degree)
        scores = cross_val_score(psvm_clf, data_set_rep, train_labels_vector,
                                 cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]
    best_MSE = min(MSE)

    optimal_degree_index = MSE.index(best_MSE)
    optimal_degree = degree_vec[optimal_degree_index]

    plt = printTunePlot('degree', degree_vec, MSE, optimal_degree, best_MSE)
    plt.show()

    return optimal_degree


### Tuning
def TrainWithTuning(data_set_rep, train_labels_vector, param_dict):
    '''
    This function preform the tuning for Hyper parameters for SVM model on all tested Kernels.
    the tune per each parameter per specific kernel is preformed with Cross Validation , K =5.
    post tuning of the unique parameter for each kernel the function tune C for all models.
    the function prints on screen the optimal value for each kernel.
    the choice of best final model post parameters tuning is set on min error on validation set.
    the function prints on screen the error for each model, and the best model who obtained min error
    :param data_set_rep: HOG representation of training set images. Prepare function output.
    :param train_labels_vector: labels array for training set. TrainTestSplit function output.
    :param param_dict: Train dictionary of parameters. GetDefaultParameters output.
    :return: SVM final model post parameters tuning
    '''
    split_ratio = param_dict['TrainValidationRatio']
    ValidationSet, TrainSetSplit, ValidationLabels, TrainLabelsSplit = \
        train_test_split(data_set_rep, train_labels_vector, test_size=split_ratio)
    print(ValidationSet.shape)
    print(TrainSetSplit.shape)
    print(ValidationLabels.shape)
    print(TrainLabelsSplit.shape)
    optimal_gamma = TuneGammaOnRBFModel(TrainSetSplit, TrainLabelsSplit)
    optimal_degree = TuneDegreeOnPolyModel(TrainSetSplit, TrainLabelsSplit)


    kernels_dict = {'linear': SVC(kernel='linear'), 'rbf': SVC(kernel='rbf', gamma=optimal_gamma),
                    'poly': SVC(kernel='poly', degree=optimal_degree)}
    models_score = {}

    for key in kernels_dict:
        clf = kernels_dict[key]
        optimal_c = TuneC(clf, TrainSetSplit, TrainLabelsSplit, key)
        clf.set_params(C=optimal_c)
        print(clf.get_params())
        clf.fit(TrainSetSplit, TrainLabelsSplit)
        validation_predictions = clf.predict(ValidationSet)
        accuracy = accuracy_score(ValidationLabels, validation_predictions)
        models_score[key] = accuracy, optimal_c

    print(models_score)
    best_model = max(models_score, key=models_score.get)
    best_accuracy = models_score[best_model]
    final_model = kernels_dict[best_model]

    print(best_model, best_accuracy)
    return final_model


def Train(data_set_rep, train_labels_vector, param_dict):
    '''
    This function Train the SVM model on the Train dataset
    :param data_set_rep: HOG representation of training set images. Prepare function output.
    :param train_labels_vector: labels array for training set. TrainTestSplit function output.
    :param param_dict: Train dictionary of parameters. GetDefaultParameters output
    :return: SVM model post training
    '''
    kernel = param_dict['Kernel']
    C = param_dict['C']
    clf = SVC(kernel=kernel, C=C).fit(data_set_rep, train_labels_vector)
    return clf


def Test(model, test_date_set):
    '''
    This function set the predictions of the Test data sets images based on trained model
    :param model: SVM model post training. Train function output.
    :param test_date_set: HOG representation of test set images. Prepare function output.
    :return: the predicted class value of test set images, and the decision function used for prediction
    test_predictions: int array including the class number predicted for each image
    decision_matrix: The decision_function method of SVC gives per-class scores for each sample
    '''
    test_predictions = model.predict(test_date_set)
    decision_matrix = model.decision_function(test_date_set)
    return test_predictions, decision_matrix


def PairOriginalTestImageIndex(max_error_indices, test_indices_tuples):
    '''
    This function return the original indices for the two largest errors images per class in Data array in order to retrieve original image
    :param max_error_indices: FindMaxErrorPerClass function output.
    :param test_indices_tuples: TrainTestSplit output
    :return: list of original image index in Data array
    '''
    original_error_indices = {}

    for key in max_error_indices.keys():
        original_image_index = []
        for value in max_error_indices[key]:
            original_image_index.append(test_indices_tuples[value][0])
        original_error_indices[key] = original_image_index

    return original_error_indices


def sortTuples(class_tuples_dict):
    '''
    This function sort the class_tuples_dict DESC in order to get the images with largest error
    :param class_tuples_dict: dictonary containing for each image its index in TestSet and margin score
    :return: class_tuples_dict sorted DESC
    '''

    for key, value in class_tuples_dict.items():
        value.sort(key=lambda x: x[1])
        error_indices = []
        #print(value)
        for result in value[0:2]:
            if result[1] == 0:
                break
            else : index_result = result[0]
            error_indices.append(index_result)
        class_tuples_dict[key] = error_indices
    #print(class_tuples_dict)
    return class_tuples_dict


def FindMaxErrorPerClass(predictions, decision_matrix, test_labels_vector, classes_info_dict):
    '''
    This funcation calculates the margin per image and return original index of the two images with highest misclassified error
    :param predictions: array of test image class prediction. Test function output.
    :param decision_matrix: The decision_function matrix of tested model.  Test function output.
    :param test_labels_vector: actual labels of the images in test data set. TrainTestSplit output.
    :param classes_info_dict:  dictionary info per class. TrainTestSplit function output.
    :return: dictonary containing per class original image index in Data array and margin score of the two images with highest misclassified error
    '''
    classes_info_key_list = (list(classes_info_dict.keys()))
    class_tuples_dict = dict((key, []) for key in classes_info_key_list)
    for i in range(len(predictions)):
        prediction_label = predictions[i]
        actual_label = test_labels_vector[i]
        actual_label_index = classes_info_key_list.index(actual_label)
        prediction_label_index = classes_info_key_list.index(prediction_label)
        #print(decision_matrix[i])
        margin = decision_matrix[i][actual_label_index] - decision_matrix[i][prediction_label_index]
        class_tuples_dict[actual_label].append((i, margin))

    return sortTuples(class_tuples_dict)


def Evaluate(predictions, decision_matrix, test_labels_vector, classes_info_dict):
    '''
    This function set the evaluation summary for the current experiment results and return it in dictonary
    :param predictions: array of test image class prediction. Test function output.
    :param decision_matrix: The decision_function matrix of tested model.  Test function output.
    :param test_labels_vector: actual labels of the images in test data set. TrainTestSplit output.
    :param classes_info_dict: dictionary info per class. TrainTestSplit function output.
    :return: dictonary of experiment results
    Error Rate : the final error rate score of the test result
    Confusion Matrix: confusion_matrix of all classes tested with 10X10 dimensions
    Max Test Error Indices: dictonary containing per class original image index in Data array of the two images with highest misclassified error
    '''
    param_summary = {'Error Rate': 1 - accuracy_score(test_labels_vector, predictions),
                     'Confusion Matrix': confusion_matrix(test_labels_vector, predictions,
                                                          labels=np.unique(test_labels_vector)),
                     'Max Test Error Indices': FindMaxErrorPerClass(predictions, decision_matrix,
                                                                    test_labels_vector, classes_info_dict)}
    return param_summary


def ReportResults(evaluation_summary, param_dict, classes_info_dict, test_indices_tuples):
    '''
    This function print the experiment test result on screen.
    :param evaluation_summary: dictonary of experiment results. Evaluate function output.
    :param param_dict: the dictionary of all experiment parameters. GetDefaultParameters output.
    :param classes_info_dict: dictionary info per class. TrainTestSplit function output.
    :param test_indices_tuples: dictonary containing per class original image index in Data array of the two images with highest misclassified error
    '''
    original_error_images_index = PairOriginalTestImageIndex(evaluation_summary['Max Test Error Indices'],
                                                             test_indices_tuples)
    print("------------Report------------")
    print("---------Classes info---------")
    for key, value in classes_info_dict.items():
        print("")
        print("Class No.", key, ":")
        print("Name of class:", value['Name'])
        print("Number of images:", value['Images'])
        print("Number of images used for training:", value['Number of images used for training'])
        print("Number of images used for testing:", value['Number of images used for testing'])
        folder_path = param_dict['Data']['Path'] + '\\' + value['Name']
        # folder_path = param_dict['Data']['Path'] + '/' + value['Name']
        original_images = sorted(os.listdir(folder_path))
        for i in range(len(original_error_images_index[key])):
            image_error_index_OG = original_error_images_index[key][i]
            image_error_index_BW = evaluation_summary['Max Test Error Indices'][key][i]
            error_images_name = original_images[image_error_index_OG]
            img_path = folder_path + '\\' + error_images_name
            # img_path = folder_path + '/' + error_images_name
            print("Error image No.", i + 1, ":", error_images_name)
            image_error_BW = SplitData['TestData'][image_error_index_BW]
            image_error_OG = cv2.imread(img_path)
            image_error_OG = cv2.cvtColor(image_error_OG, cv2.COLOR_BGR2RGB)
            #plt.imshow(image_error_BW)
            #plt.show()
            plt.imshow(image_error_OG)
            plt.show()

        print("")
        print("-----------------------------")

    print("---------Models info---------")
    print("")
    print("SVM classifiar via Hog representation")
    print("Kernel =", param_dict['Train']['Kernel'], " C =", param_dict['Train']['C'])
    print("Error Rate =", '%.3f' % evaluation_summary['Error Rate'])

    # 1
    print("Confusion Matrix:")
    print("")
    print(evaluation_summary['Confusion Matrix'])

    # 2
    classes_names = [x['Name'] for x in list(classes_info_dict.values())]
    ax = sns.heatmap(evaluation_summary['Confusion Matrix'],
                     cmap="rocket",
                     annot=True, fmt="d",
                     xticklabels=classes_names,
                     yticklabels=classes_names)
    ax.set_title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


if __name__ == '__main__':
    Params = GetDefaultParameters()
    DandL = GetData(Params['Data'])
    SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'], Params['Split'], DandL['ClassesInfo'])
    TrainDataRep = Prepare(SplitData['TrainData'], Params['Prepare'])
    # ### Tuning
    #ModelTuning = TrainWithTuning(TrainDataRep, SplitData['TrainLabels'], Params['Train'])
    Model = Train(TrainDataRep, SplitData['TrainLabels'], Params['Train'])
    TestDataRep = Prepare(SplitData['TestData'], Params['Prepare'])
    Results, DecisionMatrix = Test(Model, TestDataRep)
    Summary = Evaluate(Results, DecisionMatrix, SplitData['TestLabels'], DandL['ClassesInfo'])
    ReportResults(Summary, Params, DandL['ClassesInfo'], SplitData['TestIndicesTuples'])
