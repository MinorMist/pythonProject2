from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

# Example molecule
smiles = 'C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O'
molecule_1 = Chem.MolFromSmiles(smiles)

# Generate Morgan fingerprint
morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule_1, radius=2, nBits=2048)

# Generate RDKit topological fingerprint
topological_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(molecule_1, nBits=2048)

# Convert fingerprints to numpy arrays for comparison
import numpy as np
from rdkit import DataStructs

morgan_fp_array = np.zeros((2048,), dtype=int)
DataStructs.ConvertToNumpyArray(morgan_fp, morgan_fp_array)

topological_fp_array = np.zeros((2048,), dtype=int)
DataStructs.ConvertToNumpyArray(topological_fp, topological_fp_array)

print("Morgan Fingerprint:\n", morgan_fp_array)
print(len(morgan_fp_array))
print("Topological Fingerprint:\n", topological_fp_array)
print(len(topological_fp_array))
