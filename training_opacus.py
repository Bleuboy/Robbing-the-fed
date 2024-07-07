#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import argparse
from torch.utils.data import Subset, DataLoader
import torch
import torchvision
from collections import namedtuple
import os
import matplotlib.pyplot as plt
from attacks.analytic_attack import ImprintAttacker
from modifications.imprint import ImprintBlock
from utils.breaching_utils import *

import medmnist
from medmnist import INFO, Evaluator

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm.notebook import tqdm

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


batch_size = 8 # Number of images in the user's batch. We have a small one here for visualization purposes
EPOCHS = 5
MAX_GRAD_NORM = 1.2
NOISE_MULTIPLIER = 1.1

def train_and_evaluate(batch_size, EPOCHS, MAX_GRAD_NORM, NOISE_MULTIPLIER, model):
    import random
    random.seed(234324) # You can change this to get a new batch.

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=data_cfg_default.mean, std=data_cfg_default.std),
        ]
    )
    data_flag = 'dermamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split="train", transform=transforms, download=True, size=224)
    samples = [dataset[i] for i in random.sample(range(len(dataset)), batch_size)]
    data = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples]).flatten()


    # ### Initialize your model

    # In[ ]:


    setup = dict(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float)

    # # This could be any model:
    # model = torchvision.models.resnet18(pretrained = True)
    # # Modify the final layer to have 7 output classes
    # model.fc = torch.nn.Linear(512, 7)
    if model is None:
        model = torchvision.models.resnet18(pretrained =True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)
    if model == 'resnet18':
        model = torchvision.models.resnet18(pretrained =True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)

    # Update the number of classes attribute
    model.num_classes = 7

    loss_fn = torch.nn.CrossEntropyLoss()


    # In[ ]:


    # It will be modified maliciously:
    input_dim = data_cfg_default.shape[0] * data_cfg_default.shape[1] * data_cfg_default.shape[2]
    num_bins = 100 # Here we define number of imprint bins
    block = ImprintBlock(input_dim, num_bins=num_bins)
    model = torch.nn.Sequential(
        torch.nn.Flatten(), block, torch.nn.Unflatten(dim=1, unflattened_size=data_cfg_default.shape), model
    )
    secret = dict(weight_idx=0, bias_idx=1, shape=tuple(data_cfg_default.shape), structure=block.structure)
    secrets = {"ImprintBlock": secret}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # In[ ]:


    model = ModuleValidator.fix(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    training_set = DataClass(split="train", transform=transforms, download=True, size=224)
    subset_indices = np.arange(100)
    subset_training_set = Subset(training_set, subset_indices)
    data_loader = DataLoader(subset_training_set, batch_size=batch_size)
    # training_set = DataClass(split="train", transform=transforms, download=True, size=224)
    # data_loader = DataLoader(training_set, batch_size=batch_size)

    # add opacus here -> problem with the model structure (ImprintBlock) so do it after the imprint block
    if hasattr(model, "autograd_grad_sample_hooks"):
        del model.autograd_grad_sample_hooks

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        max_grad_norm=MAX_GRAD_NORM,
        poisson_sampling= False,
        noise_multiplier= NOISE_MULTIPLIER,
    )
    # privacy_engine = PrivacyEngine()
    # model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=data_loader,
    #     epochs = 2,
    #     target_epsilon = EPSILON,
    #     target_delta = DELTA,
    #     max_grad_norm=MAX_GRAD_NORM,
    #     poisson_sampling= False,
    #     grad_sample_mode= "hooks",
    #     #grad_sample_mode="ew",
    # )

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")


    # In[ ]:


    MAX_PHYSICAL_BATCH_SIZE = batch_size
    DELTA = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def accuracy(preds, labels):
        return (preds == labels).mean()
    def train(model, train_loader, optimizer, epoch, device):
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        criterion.to(device)
        losses = []
        top1_acc = []

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for i, (images, target) in enumerate(memory_safe_data_loader):
                optimizer.zero_grad()
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                target = target.flatten()
                loss = criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                # measure accuracy and record loss
                acc = accuracy(preds, labels)

                losses.append(loss.item())
                top1_acc.append(acc)

                loss.backward()
                optimizer.step()
                print(i)
                if (i+1) % 200 == 0:
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )


    # In[ ]:


    def test(model, test_loader, device):
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        criterion
        losses = []
        top1_acc = []

        with torch.no_grad():
            for images, target in test_loader:
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                target = target.flatten()
                loss = criterion(output, target)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)

                losses.append(loss.item())
                top1_acc.append(acc)

        top1_avg = np.mean(top1_acc)

        print(
            f"\tTest set:"
            f"Loss: {np.mean(losses):.6f} "
            f"Acc: {top1_avg * 100:.6f} "
        )
        return np.mean(top1_acc)


    # In[ ]:


    from tqdm.notebook import tqdm


    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        train(model, data_loader, optimizer, epoch + 1, device)


    # In[ ]:


    test_model = model
    test_set = DataClass(split="test", transform=transforms, download=True, size=224)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model_trained = model._modules.get('_module')

    #print(model_trained(data.to(device)))
    #print(labels)


    # ### Simulate an attacked FL protocol

    # In[ ]:


    # This is the attacker:
    attacker = ImprintAttacker(model_trained, loss_fn, attack_cfg_default, setup)

    #Server-side computation:
    queries = [dict(parameters=[p for p in model_trained.parameters()], buffers=[b for b in model_trained.buffers()])]
    server_payload = dict(queries=queries, data=data_cfg_default)

    #User-side computation:


    loss = loss_fn(model_trained(data.to(device)), labels.to(device))
    shared_data = dict(
        gradients=[torch.autograd.grad(loss, model_trained.parameters())],
        buffers=None,
        num_data_points=1,
        labels=labels,
        local_hyperparams=None,
    )
    print(loss)


    # In[ ]:


    # Attack:
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, secrets, dryrun=False)


    # In[ ]:


    # Metrics?:
    from utils.analysis import report
    true_user_data = {'data': data, 'labels': labels}
    metrics = report(reconstructed_user_data,
        true_user_data,
        server_payload,
        model, compute_ssim=False) # Can change to true and install a package...
    print(f"MSE: {metrics['mse']}, PSNR: {metrics['psnr']}, LPIPS: {metrics['lpips']}, SSIM: {metrics['ssim']} ")


    # ### Plot ground-truth data

    # In[ ]:


    plot_data(data_cfg_default, true_user_data, setup)

    # Create the "images" folder if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the images inside the "images" folder
    plt.savefig("images/true_user_data.png")


    # ### Now plot reconstructed data

    # In[ ]:


    plot_data(data_cfg_default, reconstructed_user_data, setup)
    # Save the images inside the "images" folder
    plt.savefig("images/reconstructed_user_data.png")


    # In[ ]:


    top1_acc = test(test_model, test_loader, device)

    return top1_acc, metrics['mse'], metrics['psnr'], metrics['lpips']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model with different parameters.")
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--EPOCHS', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--MAX_GRAD_NORM', type=float, required=True, help='Maximum gradient norm')
    parser.add_argument('--NOISE_MULTIPLIER', type=float, required=True, help='Noise multiplier')
    parser.add_argument('--model', type=str, required=True, help='Type of model to use')
    
    args = parser.parse_args()
    train_and_evaluate(args.batch_size, args.EPOCHS, args.MAX_GRAD_NORM, args.NOISE_MULTIPLIER, args.model)
