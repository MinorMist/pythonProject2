from django.http import HttpResponse
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rdkit.Chem.EState import EState
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from django.shortcuts import render
from rdkit import Chem
import xgboost as xgb
import pandas as pd
import numpy as np
import csv


### CALCULATING DIFFERENT MOLECULAR DESCRIPTORS ######

def calculate(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n')]

        results = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)

            if mol is None:
                results.append(None)
            else:

                # physicochemical properties
                results.append([
                    s,  # smiles
                    rdMolDescriptors.CalcMolFormula(mol),  # molecular formula
                    Descriptors.MolWt(mol),  # molecular weight
                    Descriptors.MolLogP(mol),  # Logp
                    Descriptors.TPSA(mol),  # TPSA
                    Descriptors.NumRotatableBonds(mol),  # RB
                    Descriptors.NumHAcceptors(mol),  # number of hydrogen acceptor
                    Descriptors.NumHDonors(mol),  # number of hydrogen donor
                    mol.GetNumHeavyAtoms(),  # number of heavy atom
                    Descriptors.MolMR(mol),  # molar refractivity
                    Descriptors.FractionCSP3(mol),  # fraction csp3
                    Descriptors.MolLogP(mol) - 0.74 * (mol.GetNumHeavyAtoms() ** 0.5) - 0.47,  # logD
                    (- 0.048 * (rdMolDescriptors.CalcTPSA(mol)) - 0.104 * (Descriptors.MolLogP(mol)) - 0.295),  # logS
                    (1 - Descriptors.NumRotatableBonds(mol) + Descriptors.NumHAcceptors(mol) / 10) * (
                            1 - rdMolDescriptors.CalcTPSA(mol) / 150) * (1 - Descriptors.MolLogP(mol) / 5) * (
                            1 - Descriptors.MolWt(mol) / 500),  # bioavailability score
                    Chem.rdMolDescriptors.CalcNumRings(mol),  # Number of rings
                    len([atom for atom in mol.GetAtoms() if atom.GetSymbol() != 'C']),  # Number of hetero atom
                    Chem.rdmolops.GetFormalCharge(mol),  # Formal charge
                    len([atom for atom in mol.GetAtoms() if atom.GetNumExplicitHs() == 0 and not atom.GetChiralTag()]),
                    # Number of rigid bond
                    Descriptors.TPSA(mol),  # Polar surface area
                    mol.GetNumHeavyAtoms() - mol.GetNumBonds() + 1,  # sp3 count
                    sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),  # radical electron
                    sum(atom.GetTotalValence() for atom in mol.GetAtoms()),  # valence electron
                    sum(atom.GetMass() for atom in mol.GetAtoms() if not atom.GetSymbol() == 'H'),
                    # heavy atom molecular weight
                    Chem.Descriptors.ExactMolWt(mol),  # exact molecular weight
                    QED.qed(mol),  # Quantitative Estimation of Drug-likeness

                    # Absorption

                    Descriptors.MolLogP(mol) > 0.0,  # P-glycoprotein
                    10 ** (0.022 * Descriptors.TPSA(mol) - 0.675 * Descriptors.MolLogP(mol) - 0.005 * Descriptors.MolWt(
                        mol) + 0.861),  # Human Intestinal Absorption
                    Descriptors.MolWt(mol) < 500,  # Protein Binding Percentage

                    # Distribution
                    Descriptors.MolLogP(mol) < -0.3,  # Blood-Brain Barrier
                    0.74 * Descriptors.MolLogP(mol) - 0.007 * Descriptors.MolWt(
                        mol) - 0.27 * Descriptors.NumRotatableBonds(mol) - 0.42 * Descriptors.NumHAcceptors(mol) - 1.12,
                    # Fraction unbound

                    # Lipophilicity
                    Descriptors.MolLogP(mol),  # alogp
                    Descriptors.MolLogP(mol),  # Xlogp
                    (Descriptors.MolLogP(mol) - 0.74) * (Descriptors.NumRotatableBonds(mol) - 0.007) * (
                            Descriptors.MolWt(mol) < 5000) + 0.22,  # ilogp
                    sum(EState.EStateIndices(mol)),  # wlogp

                    #Metabolism
                    Descriptors.MolWt(mol) < 450,  # Met

                    # Excretion
                    0.025 * (Descriptors.MolWt(mol)) ** 0.75 * 10 ** (0.107 * (Descriptors.MolLogP(mol))),  # clearance
                    0.025 * Descriptors.MolWt(mol) ** 0.75,  # intrinsic clearance
                    0.693 * (Descriptors.MolWt(mol)) ** 0.5 / (10 ** (0.006 * (Descriptors.MolLogP(mol))) + 1),
                    # Half life

                    # Toxicity
                    Descriptors.MolLogP(mol) > 5.0,
                    0.176 * (Descriptors.MolLogP(mol)) - 0.00358 * (Descriptors.MolWt(mol)) + 1.351,

                    # Druglikeness
                    ((Descriptors.MolWt(mol) <= 500) and (Descriptors.MolLogP(mol) <= 5) and (
                            Descriptors.NumHDonors(mol) <= 5) and (Descriptors.NumHAcceptors(mol) <= 10)),
                    # Lipinski rule

                    (Descriptors.NumRotatableBonds(mol) <= 10) and (Descriptors.TPSA(mol) <= 140) and (
                            Descriptors.NumHAcceptors(mol) <= 10),  # Veber rule

                    (160 <= (Descriptors.MolWt(mol) <= 480)) and (-0.4 <= (0.66 * (
                            Descriptors.MolLogP(mol) - 0.005 * (Descriptors.MolMR(mol) ** 2) + 0.066)) <= 5.6) and (
                            40 <= Descriptors.MolMR(mol) <= 130) and (20 <= (mol.GetNumAtoms()) <= 70),  # Ghose rule

                    ((0.66 * (Descriptors.MolLogP(mol) - 0.005 * (Descriptors.MolMR(mol) ** 2) + 0.066)) <= 5.88) and (
                            Descriptors.TPSA(mol) <= 131.6),  # Egan rule
                    ((Descriptors.MolLogP(mol) > 3 and (Descriptors.TPSA(mol) < 75))),  # pfizer rule

                ])
        return render(request, 'myapp/results.html', {'results': results})
    else:
        return render(request, 'myapp/form.html')


# download  csv
def download(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n')]

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="admet_properties.csv"'
        writer = csv.writer(response)
        writer.writerow(
            ['SMILES', 'Formula', 'MW', 'LogP', 'TPSA', 'NumRotatableBonds', 'HBA', 'HBD', 'Num_heavy_atoms',
             'MOLAR REFRACTIVITY', 'FRACTION CSP3', 'LOGD', 'LOGS', 'BIOAVAILABILITY SCORE', 'NUM_RINGS',
             'NUM_HETEROATOMS', 'FORMAL CHARGE', 'NUM_RIGIDATOMS', 'PSA', 'SP3_COUNT', 'Radical_Electron',
             'Valence_Electron', 'HeavyAtomMolWt', 'ExactMolWt', 'qed_value', 'P_GP', 'HIA', 'PPB', 'BBB',
             'FRACTION UNBOUND', 'ALOGP', 'XLOGP3', 'ILOGP', 'WLOGP','Met', 'Cl', 'CLint', 'T_HALF', 'TOX', 'BCF', 'LIPINSKI',
             'VEBER', 'GHOSE', 'EGAN ', 'PFIZER'])
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                writer.writerow([s, 'Invalid SMILES'])
            else:

                writer.writerow([s,
                                 rdMolDescriptors.CalcMolFormula(mol),
                                 Descriptors.MolWt(mol),
                                 Descriptors.MolLogP(mol),
                                 Descriptors.TPSA(mol),
                                 Descriptors.NumRotatableBonds(mol),
                                 Descriptors.NumHAcceptors(mol),
                                 Descriptors.NumHDonors(mol),
                                 mol.GetNumHeavyAtoms(),
                                 Descriptors.MolMR(mol),
                                 Descriptors.FractionCSP3(mol),
                                 Descriptors.MolLogP(mol) - 0.74 * (mol.GetNumHeavyAtoms() ** 0.5) - 0.47,
                                 (- 0.048 * (rdMolDescriptors.CalcTPSA(mol)) - 0.104 * (
                                     Descriptors.MolLogP(mol)) - 0.295),
                                 (1 - Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol) / 10) * (
                                             1 - rdMolDescriptors.CalcTPSA(mol) / 150) * (
                                             1 - Descriptors.MolLogP(mol) / 5) * (1 - Descriptors.MolWt(mol) / 500),
                                 Chem.rdMolDescriptors.CalcNumRings(mol),
                                 len([atom for atom in mol.GetAtoms() if atom.GetSymbol() != 'C']),
                                 Chem.rdmolops.GetFormalCharge(mol),
                                 len([atom for atom in mol.GetAtoms() if
                                      atom.GetNumExplicitHs() == 0 and not atom.GetChiralTag()]),
                                 Descriptors.TPSA(mol),
                                 mol.GetNumHeavyAtoms() - mol.GetNumBonds() + 1,
                                 sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
                                 sum(atom.GetTotalValence() for atom in mol.GetAtoms()),
                                 sum(atom.GetMass() for atom in mol.GetAtoms() if not atom.GetSymbol() == 'H'),
                                 Chem.Descriptors.ExactMolWt(mol),
                                 QED.qed(mol),

                                 Descriptors.MolLogP(mol) > 0.0,
                                 10 ** (0.022 * Descriptors.TPSA(mol) - 0.675 * Descriptors.MolLogP(
                                     mol) - 0.005 * Descriptors.MolWt(mol) + 0.861),

                                 Descriptors.MolWt(mol) < 500,

                                 Descriptors.MolLogP(mol) < -0.3,
                                 0.74 * Descriptors.MolLogP(mol) - 0.007 * Descriptors.MolWt(
                                     mol) - 0.27 * Descriptors.NumRotatableBonds(
                                     mol) - 0.42 * Descriptors.NumHAcceptors(mol) - 1.12,

                                 Descriptors.MolLogP(mol),
                                 Descriptors.MolLogP(mol),
                                 (Descriptors.MolLogP(mol) - 0.74) * (Descriptors.NumRotatableBonds(mol) - 0.007) * (
                                             Descriptors.MolWt(mol) < 5000) + 0.22,
                                 sum(EState.EStateIndices(mol)),
                                 Descriptors.
                                 Descriptors.MolWt(mol) < 450,


                                 0.025 * (Descriptors.MolWt(mol)) ** 0.75 * 10 ** (0.107 * (Descriptors.MolLogP(mol))),
                                 0.025 * Descriptors.MolWt(mol) ** 0.75,
                                 0.693 * (Descriptors.MolWt(mol)) ** 0.5 / (
                                             10 ** (0.006 * (Descriptors.MolLogP(mol))) + 1),

                                 Descriptors.MolLogP(mol) > 5.0,
                                 0.176 * (Descriptors.MolLogP(mol)) - 0.00358 * (Descriptors.MolWt(mol)) + 1.351,

                                 ((Descriptors.MolWt(mol) <= 500) and (Descriptors.MolLogP(mol) <= 5) and (
                                             Descriptors.NumHDonors(mol) <= 5) and (
                                              Descriptors.NumHAcceptors(mol) <= 10)),
                                 (Descriptors.NumRotatableBonds(mol) <= 10) and (Descriptors.TPSA(mol) <= 140) and (
                                             Descriptors.NumHAcceptors(mol) <= 10),
                                 (160 <= (Descriptors.MolWt(mol) <= 480)) and (-0.4 <= (0.66 * (
                                             Descriptors.MolLogP(mol) - 0.005 * (
                                                 Descriptors.MolMR(mol) ** 2) + 0.066)) <= 5.6) and (
                                             40 <= Descriptors.MolMR(mol) <= 130) and (20 <= (mol.GetNumAtoms()) <= 70),
                                 ((0.66 * (Descriptors.MolLogP(mol) - 0.005 * (
                                             Descriptors.MolMR(mol) ** 2) + 0.066)) <= 5.88) and (
                                             Descriptors.TPSA(mol) <= 131.6),
                                 ((Descriptors.MolLogP(mol) > 3 and (Descriptors.TPSA(mol) < 75)))
                                 ])
        return response


##### FINGERPRINT ########
def calculate_fingerprint(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fingerprint = [int(bit) for bit in fingerprint.ToBitString()]

        # MACCS fingerprint
        maccs_fingerprint = MACCSkeys.GenMACCSKeys(mol)
        maccs_fingerprint = [int(bit) for bit in maccs_fingerprint.ToBitString()]

        # Topological Torsion fingerprint
        torsion_fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
        torsion_fingerprint = [int(bit) for bit in torsion_fingerprint.ToBitString()]

        # RDKit fingerprint
        rdk_fingerprint = RDKFingerprint(mol)
        rdk_fingerprint = [int(bit) for bit in rdk_fingerprint.ToBitString()]

        return render(request, 'myapp/fingerprint.html',
                      {'fingerprint': fingerprint, 'smiles': smiles, 'maccs_fingerprint': maccs_fingerprint,
                       'torsion_fingerprint': torsion_fingerprint, 'rdk_fingerprint': rdk_fingerprint, })
    else:
        return render(request, 'myapp/form.html')


def download_morgan_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Calculate the Morgan fingerprint
        morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_fingerprint = [int(bit) for bit in morgan_fingerprint.ToBitString()]

        # Prepare the CSV data
        fieldnames = ['Fingerprint','Bits']
        rows = [
            ['Morgan Fingerprint'] + morgan_fingerprint,
        ]

        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="morgan_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(fieldnames)
        writer.writerows(rows)
        return response
    else:
        return render(request, 'myapp/form.html')


def download_maccs_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Calculate the MACCS fingerprint
        maccs_fingerprint = MACCSkeys.GenMACCSKeys(mol)
        maccs_fingerprint = [int(bit) for bit in maccs_fingerprint.ToBitString()]

        # Prepare the CSV data
        fieldnames = ['Fingerprint','bits']
        rows = [
            ['MACCS Fingerprint'] + maccs_fingerprint,
        ]

        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="maccs_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(fieldnames)
        writer.writerows(rows)

        return response
    else:
        return render(request, 'myapp/form.html')

import csv

def download_torsion_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Calculate the Topological Torsion fingerprint
        torsion_fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
        torsion_fingerprint = [int(bit) for bit in torsion_fingerprint.ToBitString()]


        fieldnames = ['Fingerprint', 'bits']
        rows = [
            ['Topological Torsion Fingerprint'] + torsion_fingerprint,
        ]
        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="torsion_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(fieldnames)
        writer.writerows(rows)

        return response
    else:
        return render(request, 'myapp/form.html')


def download_rdk_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Calculate the RDKit fingerprint
        rdk_fingerprint = Chem.RDKFingerprint(mol)
        rdk_fingerprint = [int(bit) for bit in rdk_fingerprint.ToBitString()]


        # Prepare the CSV data
        fieldnames = ['Fingerprint', 'bits']
        rows = [
            ['RDKit Fingerprint'] + rdk_fingerprint ,
        ]

        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="rdk_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(fieldnames)
        writer.writerows(rows)

        return response
    else:
        return render(request, 'myapp/form.html')


def home(request):
    return render(request, 'myapp/form.html')

#####  END FINGERPRINT ########


##### PREDICTION ##############
# Load the dataset containing SMILES and target values
dataset = pd.read_csv('myapp/static/CSV/final.csv')

# Separate the features (SMILES) and target values (Target) from the dataset
X = dataset['SMILES']
y = dataset['Target']


# Extract descriptors from the molecules
def extract_descriptors(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    descriptors = []
    for mol in mols:
        descriptor = []
        descriptor.append(Chem.Descriptors.MolLogP(mol))
        descriptor.append(Chem.Descriptors.MolMR(mol))
        descriptor.append(Chem.Descriptors.TPSA(mol))
        descriptor.append(Chem.Descriptors.NumRotatableBonds(mol))
        descriptor.append(Chem.Descriptors.NumHAcceptors(mol))
        descriptor.append(Chem.Descriptors.NumHDonors(mol))
        descriptor.append(Chem.Descriptors.HeavyAtomCount(mol))
        descriptor.append(Chem.Descriptors.MolWt(mol))
        descriptor.append(Chem.Descriptors.ExactMolWt(mol))
        descriptor.append(Chem.Descriptors.FractionCSP3(mol))
        descriptor.append(
            rdMolDescriptors.CalcNumRings(mol))  # Use rdMolDescriptors.NumRings(mol) for older RDKit versions

        descriptor.append(Chem.Descriptors.NumHeteroatoms(mol))
        descriptor.append(Chem.Descriptors.NumValenceElectrons(mol))
        descriptor.append(Chem.Descriptors.NumRadicalElectrons(mol))
        descriptors.append(descriptor)
    return np.array(descriptors)


# Extract features from the SMILES in the dataset
X_features = extract_descriptors(X)

# Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_features, y)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



def predict(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n')]

        # Extract features from the input SMILES
        input_features = extract_descriptors(smiles_list)

        results = []
        for i in range(len(smiles_list)):
            input_smiles = smiles_list[i]
            input_feature = input_features[i]

            # Make prediction using the XGBoost model
            prediction = model.predict(input_feature.reshape(1, -1))[0]

            # Determine the druglikeness based on the prediction
            druglikeness = "Druglike" if prediction == 1 else "Non-druglike"

            # Append the SMILES notation and druglikeness prediction to results
            results.append({'SMILES': input_smiles, 'Prediction': druglikeness, 'accuracy': accuracy})

        return render(request, 'myapp/predict.html', {'results': results})

    else:
        return render(request, 'myapp/form.html')
##### END PREDICTION ###########                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
