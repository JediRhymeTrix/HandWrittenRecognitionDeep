from collections import OrderedDict

import torch

from HWCRUtils import HWCRUtils
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep

def test_train():
    final_parameters = OrderedDict(
        lr=[0.001],
        batch_size=[50],
        shuffle=[False]
    )
    run_list = HWCRUtils.get_runs(final_parameters)
    data_set_path = "Images.npy"
    label_set_path = "Labels.npy"
    
    image_dims = (150, 150)
    epochs = 1
    split_size = 0.03
    classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

    device = HWCRUtils.get_device()

    model_directory_path = './model/model'
    save_logistics_file_path = './metrics/'
    hwRD = HandWrittenRecognitionDeep()
    X_train, Y_train, train_set, test_set, validation_set, validation_size,  validation_set_size, test_set_size = \
        hwRD.split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size, device)

    run = run_list[0]

    model_path = "model.pth"

    model = hwRD.train_model(run, train_set, model_directory_path, model_path, save_logistics_file_path, epochs, show_plot=True)
    response_test = hwRD.test_model(model, test_set, test_set_size, classes, run, device, show_confusion_matrix=True)
    print(response_test['accuracy'])


if __name__ == '__main__':
    device = HWCRUtils.get_device()
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    print("Running on " + str(device))
    test_train()
