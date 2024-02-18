from torch.utils.data import Dataset
import numpy as np
import torch as tch
import os
import json


import loader


class PreTrainECGDatasetLoader(Dataset):
    
    def __init__(self, augs=None, baseDir='', patients=[], normalize=True, normMethod='0to1', rhythmType='Rhythm', numECGstoFind=1):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.patientLookup = []
        self.augs = augs

        
        
        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients

        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        self.fileList = []
        if len(patients) != 0:
            for pat in self.patients:
                self.findEcgs(pat)
        
    
    def findEcgs(self, patient):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['validECGs']
        for ecgIx in range(numberOfEcgs):
            ecgId = str(patientInfo["ecgFileIds"][ecgIx])
            zeros = 5 - len(ecgId)
            ecgId = "0"*zeros+ ecgId
            self.fileList.append(os.path.join(patient,
                                        f'ecg_{ecgIx}',
                                        f'{ecgId}_{self.rhythmType}.npy'))
            self.patientLookup.append(patient)
            
    def __getitem__(self, item):
        if self.augs is None:
            print("Provide TwoCropsTransform object as aug")
            return 
        
        patientInfoPath = os.path.join(self.baseDir, self.patientLookup[item], 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        
        ecgData = np.load(ecgPath)

        ejectionFraction = tch.tensor(patientInfo['ejectionFraction'])
        ecgs = tch.tensor(ecgData).float()

        if self.normalize:
            if self.normMethod == '0to1':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    ecgs = ecgs - tch.min(ecgs)
                    ecgs = ecgs / tch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if tch.any(tch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return (self.augs(ecgs), ejectionFraction)
    
    def __len__(self):
        return len(self.fileList)


class ECGDatasetLoader(Dataset):
    
    def __init__(self,  baseDir='', patients=[], normalize=True, normMethod='0to1', rhythmType='Rhythm', numECGstoFind=1):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.patientLookup = []

        
        
        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients

        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        self.fileList = []
        if len(patients) != 0:
            for pat in self.patients:
                self.findEcgs(pat)
        
    
    def findEcgs(self, patient):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['validECGs']
        for ecgIx in range(numberOfEcgs):
            ecgId = str(patientInfo["ecgFileIds"][ecgIx])
            zeros = 5 - len(ecgId)
            ecgId = "0"*zeros+ ecgId
            self.fileList.append(os.path.join(patient,
                                        f'ecg_{ecgIx}',
                                        f'{ecgId}_{self.rhythmType}.npy'))
            self.patientLookup.append(patient)
            
    def __getitem__(self, item):
        
        patientInfoPath = os.path.join(self.baseDir, self.patientLookup[item], 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        
        ecgData = np.load(ecgPath)

        ejectionFraction = tch.tensor(patientInfo['ejectionFraction'])
        ecgs = tch.tensor(ecgData).float().unsqueeze(0)

        if self.normalize:
            if self.normMethod == '0to1':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    ecgs = ecgs - tch.min(ecgs)
                    ecgs = ecgs / tch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if tch.any(tch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return ecgs, ejectionFraction
    
    def __len__(self):
        return len(self.fileList)