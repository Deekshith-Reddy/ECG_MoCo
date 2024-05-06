import numpy as np
import random
import torch as tch
import torch.nn as nn
import matplotlib.pyplot as plt
import loader
import pickle
import os
from torch.utils.data import Dataset
import numpy as np
import torch as tch
import os
import json


import loader
import moco_builder
import DataTools
import Networks
import parameters

device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
gpuIds = list(range(tch.cuda.device_count()))

args = dict(
    batch_size=500,
    pretrained="/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/checkpoints/checkpoint_BaselineWarping/std_10_knot_10/"+"checkpoint_best.pth.tar",
    # pretrained="/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/checkpoints/checkpoint_MagnitudeWarping/std_0.2_knots_16/"+"checkpoint_best.pth.tar",
    perplexity=50,
)

def main(args):
    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)

    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    dataset = DataTools.PatientECGDatasetLoader(baseDir=dataDir, patients=pre_train_patients.tolist(), normalize=False)

    dataloader = tch.utils.data.DataLoader(
    dataset,
    batch_size=args["batch_size"],
    num_workers=32,
    shuffle=False,
    )

    encoder = Networks.ECG_SpatioTemporalNet
    model = moco_builder.MoCo(encoder)

    checkpoint = tch.load(args["pretrained"], map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.") and not k.startswith(
            "module.finalLayer."
        ):
            # remove prefix
            state_dict[k[len("module.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]


    msg = model.load_state_dict(state_dict, strict=True)
    print(f"There are {len(msg.missing_keys)} missing keys")
    
    model = model.encoder_q
    model = tch.nn.DataParallel(model, device_ids=gpuIds)   
    print(model)
    model.to(device)
    model.eval()

    outputs = None
    labels = None
    for i, (ecg, clinicalParam) in enumerate(dataloader):
        
        print(f"Running through batch {i} of {len(dataloader)}", end='\r')
        ecg = ecg.to(device)
        out = model(ecg).detach().cpu()
        if outputs is None:
            outputs = out
        else:
            outputs = tch.cat((outputs, out), dim=0)
        if labels is None:
            labels = clinicalParam
        else:
            labels = tch.cat((labels, clinicalParam), dim=0)

    final = outputs.cpu().numpy()

    print("Running t-SNE")
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, perplexity=args["perplexity"], learning_rate=200, n_iter=1000)
    tsne_results = tsne.fit_transform(final)

    print("Plotting")
    plt.figure(figsize=(30, 18))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, alpha=0.5, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Label value')

    # for i, label in enumerate(labels):
    #     plt.text(tsne_results[i, 0], tsne_results[i, 1], str(int(label.item())))

    plt.title('t-SNE visualization of model outputs')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('output.png')
    print("Saved output.png")

if __name__ == "__main__":
    main(args)
