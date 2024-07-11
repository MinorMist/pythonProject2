from django.http import JsonResponse
import json
from django.http import HttpResponse
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED, EState
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


def sample(request):
    return render(request, 'myapp/sample.html',)


def result(request):
    return render(request, 'myapp/result_page.html',)


def calculate_new_fingerprint(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        # mol = Chem.MolFromSmiles(smiles)

        # # Morgan fingerprint
        # fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        #     mol, radius=2, nBits=1024)
        # fingerprint = [int(bit) for bit in fingerprint.ToBitString()]

        # # MACCS fingerprint
        # maccs_fingerprint = MACCSkeys.GenMACCSKeys(mol)
        # maccs_fingerprint = [int(bit)
        #                      for bit in maccs_fingerprint.ToBitString()]

        # # Topological Torsion fingerprint
        # torsion_fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        #     mol)
        # torsion_fingerprint = [int(bit)
        #                        for bit in torsion_fingerprint.ToBitString()]

        # # RDKit fingerprint
        # rdk_fingerprint = RDKFingerprint(mol)
        # rdk_fingerprint = [int(bit) for bit in rdk_fingerprint.ToBitString()]

        # fingerprints = {'fingerprint': fingerprint, 'smiles': smiles, 'maccs_fingerprint': maccs_fingerprint,
        #                 'torsion_fingerprint': torsion_fingerprint, 'rdk_fingerprint': rdk_fingerprint, }

        return render(request, 'myapp/fingerprint_page.html', {'smiles': smiles})
    else:
        return render(request, 'myapp/fingerprint_page.html')


def calculate_new_druglikeness(request):
    return render(request, 'myapp/druglikeness_page.html')


# Molecular descriptor calculator
def calculate_new(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n') if s.split()]

        results = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                results.append(None)
            else:
                result = {
                    "smiles": s,
                    "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
                    "molecular_weight": Descriptors.MolWt(mol),
                    "logP": Descriptors.MolLogP(mol),
                    "tpsa": Descriptors.TPSA(mol),
                    "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                    "num_h_acceptors": Descriptors.NumHAcceptors(mol),
                    "num_h_donors": Descriptors.NumHDonors(mol),
                    "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                    "molar_refractivity": Descriptors.MolMR(mol),
                    "fraction_csp3": Descriptors.FractionCSP3(mol),
                    "logD": Descriptors.MolLogP(mol) - 0.74 * (mol.GetNumHeavyAtoms() ** 0.5) - 0.47,
                    "logS": -0.048 * (rdMolDescriptors.CalcTPSA(mol)) - 0.104 * (Descriptors.MolLogP(mol)) - 0.295,
                    "bioavailability_score": (1 - Descriptors.NumRotatableBonds(mol) + Descriptors.NumHAcceptors(mol) / 10) * (1 - rdMolDescriptors.CalcTPSA(mol) / 150) * (1 - Descriptors.MolLogP(mol) / 5) * (1 - Descriptors.MolWt(mol) / 500),
                    "num_rings": Chem.rdMolDescriptors.CalcNumRings(mol),
                    "num_hetero_atoms": len([atom for atom in mol.GetAtoms() if atom.GetSymbol() != 'C']),
                    "formal_charge": Chem.rdmolops.GetFormalCharge(mol),
                    "num_rigid_bonds": len([atom for atom in mol.GetAtoms() if atom.GetNumExplicitHs() == 0 and not atom.GetChiralTag()]),
                    "polar_surface_area": Descriptors.TPSA(mol),
                    "sp3_count": mol.GetNumHeavyAtoms() - mol.GetNumBonds() + 1,
                    "radical_electrons": sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
                    "valence_electrons": sum(atom.GetTotalValence() for atom in mol.GetAtoms()),
                    "heavy_atom_molecular_weight": sum(atom.GetMass() for atom in mol.GetAtoms() if not atom.GetSymbol() == 'H'),
                    "exact_molecular_weight": Chem.Descriptors.ExactMolWt(mol),
                    "qed": QED.qed(mol),
                    "ab_p_glycoprotein": "True" if Descriptors.MolLogP(mol) > 0.0 else "False",
                    "ab_human_intestinal_absorption": 10 ** (0.022 * Descriptors.TPSA(mol) - 0.675 * Descriptors.MolLogP(mol) - 0.005 * Descriptors.MolWt(mol) + 0.861),
                    "ab_protein_binding_percentage": "True" if Descriptors.MolWt(mol) < 500 else "False",
                    "ds_blood_brain_barrier": "True" if Descriptors.MolLogP(mol) < -0.3 else "False",
                    "ds_fraction_unbound": 0.74 * Descriptors.MolLogP(mol) - 0.007 * Descriptors.MolWt(mol) - 0.27 * Descriptors.NumRotatableBonds(mol) + 0.35,
                    "li_alogp": Descriptors.MolLogP(mol),
                    "li_xlogp": 0.5 * Descriptors.MolLogP(mol) + 0.3,
                    "li_ilogp": 0.42 * Descriptors.MolLogP(mol) + 0.29,
                    "li_wlogp": 0.1 * Descriptors.MolLogP(mol) + 0.04,
                    "metabolism": "True" if Descriptors.MolLogP(mol) > 3.0 else "False",
                    "ex_clearance": 10 ** (-0.278 * Descriptors.MolLogP(mol) + 0.194 * Descriptors.TPSA(mol) + 0.018 * Descriptors.NumRotatableBonds(mol) - 0.223),
                    "ex_intrinsic_clearance": 10 ** (-0.74 * Descriptors.MolLogP(mol) + 0.67 * Descriptors.TPSA(mol) + 0.045 * Descriptors.NumRotatableBonds(mol) - 0.53),
                    "ex_half_life": 0.693 / (10 ** (-0.278 * Descriptors.MolLogP(mol) + 0.194 * Descriptors.TPSA(mol) + 0.018 * Descriptors.NumRotatableBonds(mol) - 0.223)),
                    "tx_high_logP": "True" if Descriptors.MolLogP(mol) > 5.0 else "False",
                    "tx_toxicity_score": Descriptors.MolLogP(mol) * Descriptors.NumHAcceptors(mol) / Descriptors.MolWt(mol),
                    "dl_lipinski_rule": "True" if Descriptors.MolWt(mol) < 500 and Descriptors.MolLogP(mol) < 5.0 and Descriptors.NumHAcceptors(mol) < 10 and Descriptors.NumHDonors(mol) < 5 else "False",
                    "dl_veber_rule": "True" if Descriptors.TPSA(mol) < 140 and Descriptors.NumRotatableBonds(mol) < 10 else "False",
                    "dl_ghose_rule": "True" if 160 < Descriptors.MolWt(mol) < 480 and -0.4 < Descriptors.MolLogP(mol) < 5.6 and 40 < Descriptors.MolMR(mol) < 130 else "False",
                    "dl_egan_rule": "True" if Descriptors.MolLogP(mol) < 5.88 and Descriptors.TPSA(mol) < 131.6 else "False",
                    "dl_pfizer_rule": "True" if Descriptors.MolLogP(mol) < 3.0 and Descriptors.MolWt(mol) < 400 else "False",
                }
                results.append(result)

        return render(request, 'myapp/result_page.html', {'results': results})
    else:
        return render(request, 'myapp/result_page.html')

# Molecular descriptor download


def download(request):
    if request.method == 'POST':
        results_json = request.POST.get('results', '')

        try:
            results = json.loads(results_json)
        except json.JSONDecodeError:
            # Handle JSON decode error
            return HttpResponse("Invalid JSON data", status=400)

        # Initialize the response and writer
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="admet_properties.csv"'

        writer = csv.writer(response)

        # Write CSV headers
        writer.writerow([
            'SMILES', 'Molecular Formula', 'Molecular Weight', 'LogP', 'TPSA', 'Num Rotatable Bonds',
            'Num H Acceptors', 'Num H Donors', 'Num Heavy Atoms', 'Molar Refractivity', 'Fraction CSP3',
            'LogD', 'LogS', 'Bioavailability Score', 'Num Rings', 'Num Hetero Atoms', 'Formal Charge',
            'Num Rigid Bonds', 'Polar Surface Area', 'SP3 Count', 'Radical Electrons', 'Valence Electrons',
            'Heavy Atom Molecular Weight', 'Exact Molecular Weight', 'QED', 'P-Glycoprotein',
            'Human Intestinal Absorption', 'Protein Binding Percentage', 'Blood-Brain Barrier',
            'Fraction Unbound', 'AlogP', 'XlogP', 'IlogP', 'WlogP', 'Metabolism', 'Clearance',
            'Intrinsic Clearance', 'Half-Life', 'High logP', 'Toxicity Score', 'Lipinski Rule',
            'Veber Rule', 'Ghose Rule', 'Egan Rule', 'Pfizer Rule'
        ])

        # Write CSV data
        for result in results:
            writer.writerow([
                result.get('smiles', ''),
                result.get('molecular_formula', ''),
                result.get('molecular_weight', ''),
                result.get('logP', ''),
                result.get('tpsa', ''),
                result.get('num_rotatable_bonds', ''),
                result.get('num_h_acceptors', ''),
                result.get('num_h_donors', ''),
                result.get('num_heavy_atoms', ''),
                result.get('molar_refractivity', ''),
                result.get('fraction_csp3', ''),
                result.get('logD', ''),
                result.get('logS', ''),
                result.get('bioavailability_score', ''),
                result.get('num_rings', ''),
                result.get('num_hetero_atoms', ''),
                result.get('formal_charge', ''),
                result.get('num_rigid_bonds', ''),
                result.get('polar_surface_area', ''),
                result.get('sp3_count', ''),
                result.get('radical_electrons', ''),
                result.get('valence_electrons', ''),
                result.get('heavy_atom_molecular_weight', ''),
                result.get('exact_molecular_weight', ''),
                result.get('qed', ''),
                result.get('ab_p_glycoprotein', ''),
                result.get('ab_human_intestinal_absorption', ''),
                result.get('ab_protein_binding_percentage', ''),
                result.get('ds_blood_brain_barrier', ''),
                result.get('ds_fraction_unbound', ''),
                result.get('li_alogp', ''),
                result.get('li_xlogp', ''),
                result.get('li_ilogp', ''),
                result.get('li_wlogp', ''),
                result.get('metabolism', ''),
                result.get('ex_clearance', ''),
                result.get('ex_intrinsic_clearance', ''),
                result.get('ex_half_life', ''),
                result.get('tx_high_logP', ''),
                result.get('tx_toxicity_score', ''),
                result.get('dl_lipinski_rule', ''),
                result.get('dl_veber_rule', ''),
                result.get('dl_ghose_rule', ''),
                result.get('dl_egan_rule', ''),
                result.get('dl_pfizer_rule', ''),
            ])

        return response

    return HttpResponse("Method not allowed", status=405)

##### FINGERPRINT ########


def calculate_fingerprint(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=1024)
        fingerprint = [int(bit) for bit in fingerprint.ToBitString()]

        # MACCS fingerprint
        maccs_fingerprint = MACCSkeys.GenMACCSKeys(mol)
        maccs_fingerprint = [int(bit)
                             for bit in maccs_fingerprint.ToBitString()]

        # Topological Torsion fingerprint
        torsion_fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol)
        torsion_fingerprint = [int(bit)
                               for bit in torsion_fingerprint.ToBitString()]

        # RDKit fingerprint
        rdk_fingerprint = RDKFingerprint(mol)
        rdk_fingerprint = [int(bit) for bit in rdk_fingerprint.ToBitString()]

        return render(request, 'myapp/fingerprint.html',
                      {'fingerprint': fingerprint, 'smiles': smiles, 'maccs_fingerprint': maccs_fingerprint,
                       'torsion_fingerprint': torsion_fingerprint, 'rdk_fingerprint': rdk_fingerprint, })
    else:
        return render(request, 'myapp/form.html')


def download_fingerprint(request):
    if request.method == 'POST':
        fingerprint_js = request.POST.get('fingerprints', '')

        try:
            fingerprints = json.loads(fingerprint_js)
        except json.JSONDecodeError:
            # Handle JSON decode error
            return HttpResponse("Invalid JSON data", status=400)

        # Initialize the response and writer
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="fingerprint.csv"'

        writer = csv.writer(response)

        # Write CSV headers
        writer.writerow([
            'SMILE', 'Morgan fingerprint', 'Maccs fingerprint', 'Torsion fingerprint', 'Rdk fingerprint'
        ])

        # Write CSV data
        writer.writerow([fingerprints.get('smiles', ''),
                         fingerprints.get('fingerprint', ''),
                         fingerprints.get('maccs_fingerprint', ''),
                         fingerprints.get('torsion_fingerprint', ''),
                         fingerprints.get('rdk_fingerprint', '')
                         ])

        return response

    return HttpResponse("Method not allowed", status=405)


def download_new_morgan_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n') if s.split()]

        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="morgan_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(['Smiles', 'Bits'])
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            # Calculate the Morgan fingerprint
            morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=1024)
            morgan_fingerprint = [int(bit)
                                  for bit in morgan_fingerprint.ToBitString()]

            # Prepare the CSV data
            row = [s] + morgan_fingerprint
            writer.writerow(row)
        return response
    else:
        return render(request, 'myapp/fingerprint_page.html')


def download_new_maccs_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n') if s.split()]

        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="maccs_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(['Smiles', 'Bits'])
        # Calculate the MACCS fingerprint
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            maccs_fingerprint = MACCSkeys.GenMACCSKeys(mol)
            maccs_fingerprint = [int(bit)
                                 for bit in maccs_fingerprint.ToBitString()]

        # Prepare the CSV data
            row = [s] + maccs_fingerprint
            writer.writerow(row)
        return response
    else:
        return render(request, 'myapp/fingerprint_page.html')


def download_new_torsion_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n') if s.split()]

        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="torsion_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(['Smiles', 'Bits'])
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            # Calculate the Topological Torsion fingerprint
            torsion_fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol)
            torsion_fingerprint = [int(bit)
                                   for bit in torsion_fingerprint.ToBitString()]

            fieldnames = ['Fingerprint', 'bits']
            rows = [
                ['Topological Torsion Fingerprint'] + torsion_fingerprint,
            ]
            # Prepare the CSV data
            row = [s] + torsion_fingerprint
            writer.writerow(row)
        return response
    else:
        return render(request, 'myapp/fingerprint_page.html')


def download_new_rdk_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n') if s.split()]

        # Generate the CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="rdk_fingerprint.csv"'

        writer = csv.writer(response)
        writer.writerow(['Smiles', 'Bits'])
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            # Calculate the RDKit fingerprint
            rdk_fingerprint = Chem.RDKFingerprint(mol)
            rdk_fingerprint = [int(bit)
                               for bit in rdk_fingerprint.ToBitString()]
            # Prepare the CSV data
            row = [s] + rdk_fingerprint
            writer.writerow(row)
        return response
    else:
        return render(request, 'myapp/fingerprint_page.html')


# --------------------------------------
def download_morgan_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Calculate the Morgan fingerprint
        morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=1024)
        morgan_fingerprint = [int(bit)
                              for bit in morgan_fingerprint.ToBitString()]

        # Prepare the CSV data
        fieldnames = ['Fingerprint', 'Bits']
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
        maccs_fingerprint = [int(bit)
                             for bit in maccs_fingerprint.ToBitString()]

        # Prepare the CSV data
        fieldnames = ['Fingerprint', 'bits']
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


def download_torsion_csv(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        mol = Chem.MolFromSmiles(smiles)

        # Calculate the Topological Torsion fingerprint
        torsion_fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol)
        torsion_fingerprint = [int(bit)
                               for bit in torsion_fingerprint.ToBitString()]

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
            ['RDKit Fingerprint'] + rdk_fingerprint,
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
dataset = pd.read_csv('myapp/static/csv/final.csv')

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
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


def predict(request):
    if request.method == 'POST':
        smiles = request.POST.get('smiles', '')
        smiles_list = [s.strip() for s in smiles.split('\n') if s.split()]

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
            results.append(
                {'SMILES': input_smiles, 'Prediction': druglikeness, 'accuracy': accuracy})

        return render(request, 'myapp/predict.html', {'results': results})

    else:
        return render(request, 'myapp/form.html')
# END PREDICTION ###########                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
