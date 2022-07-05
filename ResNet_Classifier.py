from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import cv2
import h5py  # ### v2.10.0
import keras
import matplotlib
import matplotlib.pylab as plt  # ### matplotlib v3.3.4
import numpy as np
import platform
import random
import scipy
import scipy.io
import seaborn as sns  # ### v0.11.1
import sklearn
import tensorflow as tf
import time


random.seed(10)
np.random.seed(10)
tf.random.set_seed(10)


#################
# Main Function #
#################

def main(is_augment, is_dropout):
    ##########################################################################################
    data_path = "C:\\Users\\nbargil\\Downloads\\FlowerData-20210125\\FlowerData"  # Neta, Liel
    test_images_indices = list(range(301, 473))
    ##########################################################################################

    tic = time.time()

    is_neta_or_liel = True
    if not is_neta_or_liel:
        data_path = "C:\\Users\\ricks\\Desktop\\cv_ex2\\FlowerData"  # Ricky

    # Parameters:
    ############
    # Image loading & resizing parameters
    img_type = "jpeg"
    read_way = 1  # RGB; 3-D (if it's 0: Gray-scale; 2-D)
    input_shape = (width, height, channels) = (224, 224, 3)
    bright_factor = 50  # In case of Data Augmentation

    # Dataset preparation/ preprocessing parameter
    val_split = 0.2

    # Model parameters
    layer_to_change = -1
    dropout_factor = 0.1  # In case of Dropout
    lr = 0.00025
    optimizer = Adam(lr=lr)
    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    # Training parameters
    BATCH_SIZE = 30
    EPOCHS = 8
    callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

    # Output parameter
    num_worst = 5

    ############

    # Data preparation & pre-processing (loading, transforming, etc.)
    print("Importing data ...")

    test_images_indices = sorted(test_images_indices)
    mat = scipy.io.loadmat(data_path + "\\" + "FlowerDataLabels.mat")
    labels_array = mat['Labels'][0]
    # for i in range(labels_array.shape[0]):
    #     print("%d. '%d'" % (i + 1, labels_array[i]))
    images_indices = list(range(1, labels_array.shape[0] + 1))
    images_array = preprocess_images(data_path, images_indices, img_type, read_way, width, height,
                                     is_augment=False, bright_factor=bright_factor)

    X_test_array = images_array[[i - 1 for i in test_images_indices]]
    y_test_array = labels_array[[i - 1 for i in test_images_indices]]

    train_val_images_indices = [i for i in images_indices if i not in test_images_indices]
    X_train_val_array = images_array[[i - 1 for i in train_val_images_indices]]
    num_original_images_for_training = X_train_val_array.shape[0]
    y_train_val_array = labels_array[[i - 1 for i in train_val_images_indices]]
    num_images_true_training = sum(y_train_val_array)

    if is_augment:
        augment_images_array = preprocess_images(data_path, train_val_images_indices, img_type, read_way, width, height,
                                                 is_augment=True, bright_factor=bright_factor)
        X_train_val_array = np.append(X_train_val_array, augment_images_array, axis=0)
        y_train_val_array = np.append(y_train_val_array, y_train_val_array, axis=0)

    # Preparing train + validation + test arrays
    X_train_array, X_val_array, y_train_array, y_val_array = train_test_split(X_train_val_array,
                                                                              y_train_val_array,
                                                                              test_size=val_split,
                                                                              shuffle=True)

    print("\tDONE !")

    # lr_list = [0.000125, 0.00025, 0.0005]
    # BATCH_SIZE_list = [30, 40]
    # for lr in lr_list:
    #     for BATCH_SIZE in BATCH_SIZE_list:
    #         tic = time.time()

    # Defining model
    resnet50v2_model = ResNet50V2(include_top=False,
                                  weights='imagenet',
                                  input_shape=input_shape)
    layer = resnet50v2_model.layers[layer_to_change].output
    layer = GlobalAveragePooling2D()(layer)
    if is_dropout:
        layer = Dropout(dropout_factor)(layer)
    output = Dense(1, activation='sigmoid')(layer)  # Binary classification
    model = Model(inputs=resnet50v2_model.input, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()  # Printing network architecture

    # Training
    results = model.fit(X_train_array, y_train_array,
                        validation_data=(X_val_array, y_val_array),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[callback]
                        # ,
                        # verbose=0
                        )

    # Testing
    print("Testing ...")
    _, def_acc = model.evaluate(x=X_test_array, y=y_test_array)  # Accuracy [%] for default threshold (0.5)

    y_pred_prob_array = model.predict(x=X_test_array)
    y_pred_prob_array = y_pred_prob_array.flatten()

    # Getting best threshold
    precision, recall, thresholds = precision_recall_curve(y_test_array, y_pred_prob_array)
    num_true_ones = (y_test_array == 1).sum()
    num_true_zeros = (y_test_array == 0).sum()
    tp = recall * num_true_ones
    fp = (tp / precision) - tp
    tn = num_true_zeros - fp
    acc = (tp + tn) / (num_true_ones + num_true_zeros)
    best_acc = np.amax(acc)
    best_thresh = thresholds[np.argmax(acc)]

    y_def_pred_array = np.where(y_pred_prob_array >= 0.5, 1, 0)  # For default threshold = 0.5
    y_best_pred_array = np.where(y_pred_prob_array >= best_thresh, 1, 0)  # For selected threshold
    print("\tDONE !")

    # Report
    # Training images numbers:
    print("======")
    print("Report:")
    print("======")
    print("%s: %i, %s: %i" % ('No. of original images used for training', num_original_images_for_training,
                              'No. of original True flowers', num_images_true_training))
    print("%s: %i, %s: %i" % ('No. of augmented images', X_train_val_array.shape[0] - num_original_images_for_training,
                              'No. of augmented True flowers', sum(y_train_val_array) - num_images_true_training))
    print("%s: %i, %s: %i" % ('Total No. of images used for training', X_train_val_array.shape[0],
                              'Total No. of True flowers', sum(y_train_val_array)))
    print()
    print("Parameters:")
    print("----------")

    if is_augment:
        print("Augmentation used: YES")
    else:
        print("Augmentation used: NO")
    if is_dropout:
        print("Dropout used: YES, Dropout Value = %.2f" % dropout_factor)
    else:
        print("Dropout used: NO")
    print("Learning Rate: %.6f, Batch Size: %d" % (lr, BATCH_SIZE))

    print()
    print("y_test Actual:")
    print("-------------")
    print(y_test_array)
    print("y_test Predicted (for default threshold: 0.5):")
    print("---------------------------------------------")
    print(y_def_pred_array)
    print("y_test Predicted (for best threshold: %.3f):" % best_thresh)
    print("--------------------------------------------")
    print(y_best_pred_array)

    print()
    # print("%s: %.3f" % ('Test Loss', loss))
    print("%s: %.3f [%%]" % ('Error Rate for default threshold (0.5)', 100 - def_acc * 100))
    print("-----------------------------------------")
    print("%s (%.3f): %.3f [%%]" % ('Error Rate for best threshold', best_thresh, 100 - best_acc * 100))
    print("----------------------------------------")

    # Error images:
    print()
    print("Error test images for default threshold (0.5):")
    print("---------------------------------------------")
    # Type 1: FP
    find_worst_images(y_test_array, y_def_pred_array, y_pred_prob_array, 1, num_worst,
                      test_images_indices, img_type)

    print()
    # Type 2: FN
    find_worst_images(y_test_array, y_def_pred_array, y_pred_prob_array, 2, num_worst,
                      test_images_indices, img_type)

    print()

    toc = time.time() - tic
    print("%s: %d min, %d sec" % ('Running Time', int(toc // 60), round((toc / 60 - toc // 60) * 60)))
    print("------------")

    # Graphs:
    fig, axs = plt.subplots(1, 2)
    x = list(range(1, len(results.history['val_loss']) + 1))  # Epochs' axis (until stopping)

    # print("Validation Error Rate:  %.3f [%%]" % ((1 - results.history['val_accuracy'][-1]) * 100))
    # print("---------------------")
    # print()

    # Loss graph during training
    axs[0].plot(x, results.history['loss'])
    axs[0].plot(x, results.history['val_loss'])
    axs[0].set(xlabel='Epoch #', ylabel='Loss', title='Loss During Training')
    axs[0].legend(['Train', 'Validation'], loc='upper right')

    # Error rate graph during training
    axs[1].plot(x, [100 - y * 100 for y in results.history['accuracy']])
    axs[1].plot(x, [100 - y * 100 for y in results.history['val_accuracy']])
    axs[1].set(xlabel='Epoch #', ylabel='Error Rate [%]', title='Error Rate [%] During Training')
    axs[1].legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

    # Precision-Recall curve
    area = auc(recall, precision)
    plt.plot(recall, precision, label="auc=" + str(area))
    plt.legend(loc=4)
    plt.title("Precision-Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

    # ROC curve
    # fpr, tpr, _ = roc_curve(y_test_array, y_pred_prob_array)
    # auc = roc_auc_score(y_test_array, y_pred_prob_array)
    # plt.plot(fpr, tpr, label="auc=" + str(auc))
    # plt.legend(loc=4)
    # plt.title("ROC Curve")
    # plt.show()

    # Confusion matrix for default threshold = 0.5
    print_confusion_matrix(y_test_array, y_def_pred_array, 0.5)

    # Confusion matrix for best threshold
    print_confusion_matrix(y_test_array, y_best_pred_array, best_thresh)

#################


def verify_versions():
    """
    This function prints on screen the required and installed packages
    """
    print("Verify versions:")
    print("---------------")
    print("python REQUIRED: 3.7.5 INSTALLED:", platform.python_version())
    print("tensorflow REQUIRED 2.0.0 INSTALLED:", tf.version.VERSION)
    print("keras REQUIRED 2.3.1 INSTALLED:", keras.__version__)
    print("numpy REQUIRED 1.18.1 INSTALLED:", np.__version__)
    print("scipy REQUIRED 1.4.1 INSTALLED:", scipy.__version__)
    print("cv2 REQUIRED 4.1.2 INSTALLED:", cv2.getVersionString())
    print("sklearn REQUIRED 0.22.1 INSTALLED:", sklearn.__version__)
    print("*** Other Installed Packages:")
    print("h5py 2.10.0 CHECK:", h5py.__version__)
    print("matplotlib 3.3.4 CHECK:", matplotlib.__version__)
    print("seaborn 0.11.1 CHECK:", sns.__version__)


#######################
# Auxiliary Functions #
#######################

def preprocess_images(data_path, images_indices, img_type, read_way, width, height, is_augment, bright_factor):
    """
    This function is used for preprocess images: resize, loading and augmentation
    :param data_path: string contains directory path of the images file
    :param images_indices: list contains images indices to augment
    :param img_type: string indicates image format
    :param read_way: integer indicates reading image in RGB format or Grey Scale
    :param width: integer indicates image width for resize
    :param height: integer indicates image height for resize
    :param is_augment: boolean indicates True for augmentation
    :param bright_factor: integer indicates bright factor for augmentation
    :return: 3D array of processed images
    """
    images_array = []
    for i in images_indices:
        img_path = data_path + "\\" + str(i) + "." + img_type
        img = cv2.imread(img_path, read_way)  # Loading image
        img = cv2.resize(img, (width, height))  # Resizing image
        if is_augment:
            aug = random.randint(0, 2)
            if aug == 0 or aug == 2:  # Horizontal Flip
                img = cv2.flip(img, read_way)
            if aug == 1 or aug == 2:  # Increase Brightness
                bright = np.ones(img.shape, dtype="uint8") * bright_factor
                img = cv2.add(img, bright)
        img = img / 255.0  # Normalization
        images_array.append(np.asarray(img))
    images_array = np.array(images_array)
    return images_array


def find_worst_images(y_test_array, y_pred_array, y_pred_prob_array, err_type, num_worst,
                      test_images_indices, img_type):
    """
    This function prints on screen the worst Type-1 (FP) or Type-2 (FN) errors from the images misclassified in the test set
    :param y_test_array: array containing actual test labels
    :param y_pred_array: array containing predicted test labels by network
    :param y_pred_prob_array: array containing probability giving by the network of each image to contain flower
    :param err_type: integer indicates if Type-1 (FP) or Type-2 (FN) error
    :param num_worst: integer indicates the number of worst errors to capture
    :param test_images_indices: list containing actual test images indices at images directory
    :param img_type: string indicates image format
    """
    y_diff_array = y_test_array - y_pred_array
    y_abs_diff_prob_array = np.absolute(y_test_array - y_pred_prob_array)
    if err_type == 1:  # Type-1: FP
        err_indices = np.where(y_diff_array == -1)
        not_err_indices = np.where(y_diff_array != -1)
    else:  # Type-2: FN
        err_indices = np.where(y_diff_array == 1)
        not_err_indices = np.where(y_diff_array != 1)
    if err_indices[0].size != 0:
        if err_indices[0].size < num_worst:
            num_worst = err_indices[0].size
        y_abs_diff_prob_array[not_err_indices] = 0
        worst_abs_diff_prob_indices = (-y_abs_diff_prob_array).argsort()[:num_worst]
        print("%d worst Type-%d error images:" % (num_worst, err_type))
        print("---------------------------")
        for i in range(num_worst):
            print("%d." % (i + 1))
            print("Image: %d.%s" % (test_images_indices[worst_abs_diff_prob_indices[i]], img_type))
            print("Classification score: %.3f" % (y_pred_prob_array[worst_abs_diff_prob_indices[i]]))


def print_confusion_matrix(actual, predicted, threshold):
    """
    This function prints on screen the confusion matrix
    :param actual: array containing actual test labels
    :param predicted: array containing predicted test labels by network
    :param threshold: float number in [0,1] indicates the classification threshold
    """
    cm = confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm,
                     cmap="rocket",
                     annot=True, fmt="d",
                     xticklabels=np.unique(predicted),
                     yticklabels=np.unique(actual))
    ax.set_title("Confusion Matrix for Threshold (%.3f)" % threshold)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


##########
# Driver #
##########

if __name__ == "__main__":

    # # Optional: Checking versions
    # # ---------------------------
    # verify_versions()

    # Running configuration:
    # ---------------------
    # # Hyper Parameters Tuning
    # print("Parameters' Tuning:")
    # print("==================")

    # main(is_augment=False, is_dropout=False)  # Baseline
    # main(is_augment=True, is_dropout=False)  # Data Augmentation
    # main(is_augment=False, is_dropout=True)  # Dropout
    main(is_augment=True, is_dropout=True)  # Data Augmentation + Dropout
