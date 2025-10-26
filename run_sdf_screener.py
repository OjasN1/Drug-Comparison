import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# global variables
REFERENCE_FOLDER = 'reference_compounds'
LIBRARY_FOLDER = 'screening_library'
OUTPUT_FILE = 'screening_results.csv'
SIMILARITY_THRESHOLD = 0.5

def mol_to_fingerprint(mol):
    """Converts an RDKit molecule object into a Morgan fingerprint."""
    if mol is None:
        return None
    try:
        # Generate Morgan fingerprint (similar to ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return fp
    except Exception as e:
        print(f"Error generating fingerprint: {e}")
        return None


def load_molecule(file_path):
    """Loads a molecule from .sdf or .mol file."""
    if file_path.endswith('.sdf'):
        # Here we used SDMolSupplier for SDF files (which can contain multiple molecules) so this "supplier" is used to get them one by one.
        suppl = Chem.SDMolSupplier(file_path)
        mol = next(suppl, None)  # Get first molecule or None
    elif file_path.endswith('.mol'):
        # Use MolFromMolFile for MOL files
        mol = Chem.MolFromMolFile(file_path)
    else:
        print(f"Skipping unknown file type: {file_path}")
        return None

    return mol


print("Starting molecular structure screening...")

# 1. Load Reference Molecules and generate fingerprints
reference_data = {}  # Dictionary to store {name: fingerprint}
print(f"Loading reference molecules from '{REFERENCE_FOLDER}'...")

for filename in os.listdir(REFERENCE_FOLDER):
    file_path = os.path.join(REFERENCE_FOLDER, filename)

    # Use filename (without extension) as the disease/category name
    ref_name = os.path.splitext(filename)[0]

    mol = load_molecule(file_path)

    if mol:
        fp = mol_to_fingerprint(mol)
        if fp:
            reference_data[ref_name] = fp
            print(f"  > Loaded reference: {ref_name}")
    else:
        print(f"  ! Warning: Could not load reference molecule from {filename}")

if not reference_data:
    print("Error: No reference molecules were loaded. Exiting.")
    exit()

print(f"\nSuccessfully loaded {len(reference_data)} reference fingerprints.")

# 2. Process Screening Library
results = []
print(f"Loading and screening library from '{LIBRARY_FOLDER}'...")

for filename in os.listdir(LIBRARY_FOLDER):
    file_path = os.path.join(LIBRARY_FOLDER, filename)
    mol = load_molecule(file_path)

    if not mol:
        print(f"  ! Warning: Could not load library molecule {filename}")
        continue

    lib_fp = mol_to_fingerprint(mol)

    if not lib_fp:
        print(f"  ! Warning: Could not generate fingerprint for {filename}")
        continue

    # 3. Compare library molecule to all references
    best_score = 0.0
    best_match_name = "No Match"
    scores = {}

    for ref_name, ref_fp in reference_data.items():
        # Calculate Tanimoto Similarity
        score = DataStructs.TanimotoSimilarity(lib_fp, ref_fp)
        scores[f"Score_{ref_name}"] = score  # Store score for each category

        if score > best_score:
            best_score = score
            best_match_name = ref_name  # Store the name of the best match

    # --- NEW LOGIC: Assign category based on threshold ---
    final_category = 'Other'  # Default to 'Other'
    if best_score >= SIMILARITY_THRESHOLD:
        final_category = best_match_name  # If it's good enough, assign the best match
    # --- END NEW LOGIC ---

    # Store results for this molecule
    result_row = {
        'Molecule_File': filename,
        'Assigned_Category': final_category,  # This is the new category
        'Best_Score': best_score,  # This is the score of the best match
        'Best_Match_Found': best_match_name  # This shows the best match, even if it was below the threshold
    }
    result_row.update(scores)  # Add all individual scores
    results.append(result_row)

print(f"\nScreening complete. Processed {len(results)} molecules.")

# 4. Save results to CSV
if results:
    df = pd.DataFrame(results)

    # Re-order columns to be cleaner
    cols = ['Molecule_File', 'Assigned_Category', 'Best_Score', 'Best_Match_Found'] + [col for col in df.columns if col.startswith('Score_')]
    df = df[cols]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to '{OUTPUT_FILE}'")
else:
    print("No molecules were processed from the library.")