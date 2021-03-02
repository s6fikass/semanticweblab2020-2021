import torch
import torch.nn as nn
import torch.utils
from workflow.NN_workflow import NN_workflow


def main():
    """ Testing model defined according to ontology. """

    # Step 1: Define the layers.
    layers = [nn.Conv2d(1, 32, 3, 1),
              nn.ReLU(),
              nn.Conv2d(32, 64, 3, 1),
              nn.ReLU(),
              nn.MaxPool2d(2),
              nn.Dropout(0.25),
              nn.Flatten(),
              nn.Linear(9216, 128),
              nn.ReLU(),
              nn.Dropout(0.5),
              nn.Linear(128, 10),
              nn.LogSoftmax(1)]

    # Step 2: Initialise the criterion.
    criterion = nn.CrossEntropyLoss()

    # Step 3: Initialise the NN_workflow based on the layers defined by user and criterion.
    neural_net_workflow = NN_workflow(criterion, layers=layers)

    # Step 4: Initialise the optimizer. If this step is skipped, default optimizer SGD is used.
    optimizer = torch.optim.SGD(neural_net_workflow.model.parameters(), lr=0.001)

    # Step 5: Set the optimizer in the NN workflow. If Step 4 is skipped, this step should be skipped too.
    neural_net_workflow.set_optimizer(optimizer)

    # Step 6: Data preparation based on the dataset.
    train_loader, test_loader = neural_net_workflow.data_preparation(dset_name="MNIST")

    # Step 7: Call the train method of the NN workflow.
    neural_net_workflow.train(train_loader)

    # Step 8: Call the test method of the NN workflow.
    neural_net_workflow.test(test_loader)


if __name__ == "__main__":
    main()
