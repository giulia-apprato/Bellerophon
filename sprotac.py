import streamlit as st
import pandas as pd
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from IPython.display import HTML
import requests
IPythonConsole.ipython_useSVG = True
PandasTools.RenderImagesInAllDataFrames(images=True)

# Get smiles from mol object
def mol_to_smiles(mol):
    if mol:
        return Chem.MolToSmiles(mol)
    return None

# Get mol from smiles 
def mol_from_smiles(mol):
    if mol:
        return Chem.MolFromSmiles(mol)
    return None

# Load SDF files into DataFrames with all key fields extracted directly
def load_and_process_sdf(filename):
    df = PandasTools.LoadSDF(filename, molColName='Mol')

    # Drop rows missing essential data
    df = df.dropna(subset=['Mol'])

    # Remove duplicates by Mol
    df = df.drop_duplicates(subset='Mol', keep='first')

    return df


# Load and process the SDF files
warhead_df = load_and_process_sdf('default_warhead.sdf')
warhead_df = warhead_df.dropna(subset=['Mol'])
e3_df = load_and_process_sdf('default_E3ligand.sdf')
e3_df = e3_df.dropna(subset=['Mol'])


# Error logging function
def log_error(protac_name, error_type, smiles, details, error_messages=None):
    """
    Log an error internally
    
    Parameters:
        protac_name (str): Name of the PROTAC
        error_type (str): Short description of error
        smiles (str): The SMILES string associated with the error
        details (str): Additional context or exception message
        error_messages (list): list to store all errors
    """
    # Always record internally
    if error_messages is not None:
        error_messages.append((protac_name, error_type, smiles, details))

# Define the splitting function
def split_protac(protac_name, protac_smiles, protac_mol, warhead_df, e3_df, error_messages=None):
    """
    Splits a single PROTAC (given as RDKit mol) into warhead, linker and E3 ligand candidates
    using the provided warhead/e3 libraries. Returns a DataFrame with valid solutions (may be empty).
    """
    # Safety: if protac mol not provided, return empty DF
    if protac_mol is None:
            log_error(protac_name, "Could not parse PROTAC SMILES", protac_smiles,
                      "RDKit returned None", error_messages)

    results = []
    results_df1 = []
    no_match = []

    # --- WARHEAD MATCHING ---
    protac_matches = []
    warhead_matched = False

    for _, warhead_row in warhead_df.iterrows():
        if 'Mol' not in warhead_row or warhead_row['Mol'] is None:
            continue

        warhead_mol = warhead_row['Mol']
        if not isinstance(warhead_mol, Chem.Mol):
            continue

        try:
            matches = protac_mol.GetSubstructMatches(warhead_mol)
        except Exception as e:
                log_error(protac_name, "Sanitization failed", protac_smiles, str(e),
                          error_messages)
                continue

        if matches:
            w_name = warhead_row.get('Compound ID', None)
            for match in matches:
                linker_after_w = AllChem.DeleteSubstructs(protac_mol, warhead_mol)
                protac_matches.append({
                    'Protac Mol': protac_mol,
                    'Warhead Mol': warhead_mol,
                    'Linker Mol': linker_after_w,
                    'protac ID': protac_name,
                    'warhead ID': w_name,
                    'Protac SMILES': mol_to_smiles(protac_mol),
                    'Warhead SMILES': mol_to_smiles(warhead_mol)
                })
            warhead_matched = True

    if not warhead_matched:
        log_error(protac_name, "Warhead matching failed", protac_smiles,
                    "No substructure matches found", error_messages)
        return pd.DataFrame()

    # Save warhead matches into results list
    results.extend(protac_matches)

    # --- E3 MATCHING ---
    e3_matched = False
    for _, e3_row in e3_df.iterrows():
        e3_mol = e3_row.get('Mol', None)
        if e3_mol is None:
            continue

        try:
            matches = protac_mol.GetSubstructMatches(e3_mol)
        except Exception as e:
                log_error(protac_name, "E3 ligand matching failed", protac_smiles, str(e),
                          error_messages)
                continue

        if matches:
            e3_name = e3_row.get('Compound ID', None)
            for match in matches:
                linker_after_e3 = AllChem.DeleteSubstructs(protac_mol, e3_mol)
                results_df1.append({
                    'Protac Mol': protac_mol,
                    'E3 Mol': e3_mol,
                    'Linker Mol': linker_after_e3,
                    'E3 ID': e3_name,
                    'E3 SMILES': mol_to_smiles(e3_mol)
                })
            e3_matched = True

    if not e3_matched:
        no_match.append({
            'Protac Mol': protac_mol,
            'protac ID': protac_name,
            'no warhead match': False,
            'no e3 match': True
        })
        log_error(protac_name, "E3 ligand matching failed", protac_smiles,
                    "No substructure matches found", error_messages)
        return pd.DataFrame()

    # --- Build DataFrames and merge ---
    results_df = pd.DataFrame(results)
    results_df1 = pd.DataFrame(results_df1)

    if results_df.empty or results_df1.empty:
        return pd.DataFrame()

    # Merge using Protac Mol (will produce Linker Mol_x from warhead deletion and Linker Mol_y from E3 deletion)
    final_df = pd.merge(results_df, results_df1, on='Protac Mol', how='inner')

    if final_df.empty:
        return pd.DataFrame()

    # Compute Final Linker Mol by removing E3 substructure from Linker_x (if present)
    final_df['Final Linker Mol'] = final_df.apply(
        lambda row: (AllChem.DeleteSubstructs(row['Linker Mol_x'], row['E3 Mol'])
                     if row['E3 Mol'] is not None and row['Linker Mol_x'] is not None and
                        row['Linker Mol_x'].HasSubstructMatch(row['E3 Mol'])
                     else None),
        axis=1
    )

    # Compute Linker Mol Check by removing Warhead from Linker_y
    final_df['Linker Mol Check'] = final_df.apply(
        lambda row: (AllChem.DeleteSubstructs(row['Linker Mol_y'], row['Warhead Mol'])
                     if row['Warhead Mol'] is not None and row['Linker Mol_y'] is not None and
                        row['Linker Mol_y'].HasSubstructMatch(row['Warhead Mol'])
                     else None),
        axis=1
    )

    # Verify final linker equality (as SMILES) to ensure consistent decomposition
    final_df['Linker_check'] = final_df.apply(
        lambda row: True if row['Final Linker Mol'] is not None and row['Linker Mol Check'] is not None and
        mol_to_smiles(row['Final Linker Mol']) == mol_to_smiles(row['Linker Mol Check'])
        else False,
        axis=1
    )

    # Capture linker inconsistency before filtering
    inconsistent_rows = final_df[final_df['Linker_check'] == False]
    for _, r in inconsistent_rows.iterrows():
        log_error(
            protac_name,
            "Linker inconsistent",
            protac_smiles,
            "Linker obtained from Warhead- and E3-removal steps do not match",
            error_messages
        )
    # Keep only those that passed the linker consistency check
    final_df = final_df[final_df['Linker_check'] == True].copy()

    if final_df.empty:
        return pd.DataFrame()

    # Convert Final Linker to canonical SMILES and re-create Mol to avoid RDKit inconsistencies
    final_df['Linker SMILES'] = final_df['Final Linker Mol'].apply(mol_to_smiles)
    final_df['Final Linker Mol'] = final_df['Linker SMILES'].apply(mol_from_smiles)

    # Count disconnected fragments and keep only single-fragment linkers
    def count_disconnected_fragments(mol):
        if mol:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            return len(frags)
        return 0

    final_df['Num_Disconnected_Fragments'] = final_df['Final Linker Mol'].apply(count_disconnected_fragments)
    # Identify multifragment linkers BEFORE filtering
    multi_frag = final_df[final_df['Num_Disconnected_Fragments'] > 1]
    for _, r in multi_frag.iterrows():
        log_error(
            protac_name,
            "Multifragment linker",
            protac_smiles,
            f"Linker contains {r['Num_Disconnected_Fragments']} disconnected fragments",
            error_messages
        )

    final_df = final_df[final_df['Num_Disconnected_Fragments'] == 1].copy()

    if final_df.empty:
        return pd.DataFrame()

    # Heavy atoms and ring counts
    def cal_num_hatoms(mol):
        if mol:
            return rdMolDescriptors.CalcNumHeavyAtoms(mol)
        return None

    def ring_count(mol):
        if mol:
            return rdMolDescriptors.CalcNumRings(mol)
        return None

    final_df['Protac heavy atoms'] = final_df['Protac Mol'].apply(cal_num_hatoms)
    final_df['Warhead heavy atoms'] = final_df['Warhead Mol'].apply(cal_num_hatoms)
    final_df['E3 heavy atoms'] = final_df['E3 Mol'].apply(cal_num_hatoms)
    final_df['Linker heavy atoms'] = final_df['Final Linker Mol'].apply(cal_num_hatoms)

    final_df['Sum of heavy atoms'] = final_df[['Warhead heavy atoms', 'E3 heavy atoms', 'Linker heavy atoms']].sum(axis=1)
    final_df['Heavy atom check'] = final_df.apply(
        lambda r: (r['Sum of heavy atoms'] is not None and r['Protac heavy atoms'] is not None and
                   r['Sum of heavy atoms'] == r['Protac heavy atoms']), axis=1)
    
    # Log heavy atom errors BEFORE filtering
    heavy_atom_fail = final_df[final_df['Heavy atom check'] == False]
    for _, r in heavy_atom_fail.iterrows():
        log_error(
            protac_name,
            "Heavy atom mismatch",
            protac_smiles,
            (
                f"Protac heavy atoms = {r['Protac heavy atoms']}, "
                f"sum of components = {r['Sum of heavy atoms']}"
            ),
            error_messages
        )
    final_df = final_df[final_df['Heavy atom check'] == True].copy()

    if final_df.empty:
            return pd.DataFrame()

    # Ring counts
    final_df['Protac ring count'] = final_df['Protac Mol'].apply(ring_count)
    final_df['Warhead ring count'] = final_df['Warhead Mol'].apply(ring_count)
    final_df['E3 ring count'] = final_df['E3 Mol'].apply(ring_count)
    final_df['Linker ring count'] = final_df['Final Linker Mol'].apply(ring_count)

    final_df['Sum of ring count'] = final_df[['Warhead ring count', 'E3 ring count', 'Linker ring count']].sum(axis=1)
    final_df['Ring count check'] = final_df.apply(
        lambda r: (r['Sum of ring count'] is not None and r['Protac ring count'] is not None and
                   r['Sum of ring count'] == r['Protac ring count']), axis=1)
    # Log ring count errors BEFORE filtering
    ring_fail = final_df[final_df['Ring count check'] == False]
    for _, r in ring_fail.iterrows():
        log_error(
            protac_name,
            "Ring count mismatch",
            protac_smiles,
            (
                f"Protac rings = {r['Protac ring count']}, "
                f"sum of components = {r['Sum of ring count']}"
            ),
            error_messages
        )
    final_df = final_df[final_df['Ring count check'] == True].copy()

    if final_df.empty:
            return pd.DataFrame()

    # Add Warhead/E3 SMILES if not present
    if 'Warhead SMILES' not in final_df.columns:
        final_df['Warhead SMILES'] = final_df['Warhead Mol'].apply(mol_to_smiles)
    if 'E3 SMILES' not in final_df.columns:
        final_df['E3 SMILES'] = final_df['E3 Mol'].apply(mol_to_smiles)

    # Deduplicate identical (Warhead, Linker, E3) solutions — keep first
    final_df = final_df.drop_duplicates(subset=['Warhead SMILES', 'Linker SMILES', 'E3 SMILES'], keep='first').reset_index(drop=True)

    # Optionally attach library Compound ID info (if present)
    f_warhead_df = warhead_df[['Compound ID']].copy() if 'Compound ID' in warhead_df.columns else pd.DataFrame()
    f_e3_df = e3_df[['Compound ID']].copy() if 'Compound ID' in e3_df.columns else pd.DataFrame()

    # Merge to add Compound IDs back (if available)
    if not f_warhead_df.empty and 'warhead ID' in final_df.columns:
        final_df = pd.merge(final_df, f_warhead_df, left_on='warhead ID', right_on='Compound ID', how='left', suffixes=('', '_warhead'))
    if not f_e3_df.empty and 'E3 ID' in final_df.columns:
        final_df = pd.merge(final_df, f_e3_df, left_on='E3 ID', right_on='Compound ID', how='left', suffixes=('', '_e3'))

    # Keep only useful columns (you can expand as needed)
    cols_keep = ['Protac Mol', 'Protac SMILES', 'Warhead Mol', 'Warhead SMILES', 'E3 Mol', 'E3 SMILES',
                 'Final Linker Mol', 'Linker SMILES', 'warhead ID', 'E3 ID']
    final_df = final_df[[c for c in cols_keep if c in final_df.columns]].copy()

    return final_df

#Streamlit app--------------------------------------------------------------
import io
import datetime
import streamlit.components.v1 as components
def main():
    st.image("bellerophon_GA.svg") 
    st.write("")
    st.markdown(
    '<div style="text-align: justify">'
    'Bellerophon is a computational tool designed to automatically split PROTACs into their three components: warhead, linker, and E3 ligase ligand.<br><br>'
    'It identifies the warhead and E3 ligand from a curated library, either the default one provided by the authors or a user-uploaded version. Users can input PROTACs by pasting the name and SMILES or uploading a file.<br><br>'
    'Bellerophon supports data curation and rational design by enabling the comparison and recombination of validated building blocks for efficient PROTAC discovery.'
    ' For additional details on the methodology and instructions for using the tool, please refer to the '
    '<a href="https://github.com/giulia-apprato/Bellerophon" target="_blank" style="color:#1f77b4; text-decoration:none; font-weight:bold;">README file on GitHub</a>.'
    '</div>',
    unsafe_allow_html=True
)
    st.write("")
    st.markdown(
    """
    <hr style="border: 1px solid #ccc; margin: 25px 0;">
    """,
    unsafe_allow_html=True,
)

    # Initialize error messages list
    error_messages = []

    # Initialize session state variables if not present
    if "clean_results" not in st.session_state:
        st.session_state["clean_results"] = []
    if "error_messages" not in st.session_state:
        st.session_state["error_messages"] = []
    if "processing_done" not in st.session_state:
        st.session_state["processing_done"] = False

    # --- User email input ---
    #st.markdown("### 📧 User Information")
    #user_email = st.text_input("Please enter your email address before processing:")

    # Simple email validation
    #def is_valid_email(email):
    #    import re
    #    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    #    return re.match(pattern, email) is not None

    #if user_email and not is_valid_email(user_email):
    #    st.error("Please enter a valid email address (e.g., name@example.com) before starting the process.")

    # Proceed only if a valid email is provided
    #can_process = user_email and is_valid_email(user_email)

    # --- Skip email requirement ---
    can_process = True # Comment on this line if re-enabling email requirement

    # --- Library Selection Section ---
    st.markdown("### 📚 Library Selection")

    use_custom_libs = st.checkbox(
        "Use custom warhead and E3 ligase libraries instead of the default ones"
    )

    if use_custom_libs:
        st.markdown(
            """
            **Upload your libraries:**  
            Custom libraries must be in `.sdf` format and include the mol field.
            """
        )

        warhead_file = st.file_uploader("📤 Upload Warhead Library (.sdf)", type=["sdf"])
        e3_file = st.file_uploader("📤 Upload E3 Ligase Library (.sdf)", type=["sdf"])
    else:
        st.info("Default warhead and E3 ligase libraries will be used.")

    # --- Helper function for validation ---
    def validate_library(df, library_type):
        """Ensure that uploaded SDF contains required columns."""
        missing_cols = [col for col in ["Name", "Smiles"] if col not in df.columns]
        if missing_cols:
            st.error(
                f"Invalid {library_type} SDF format. Missing column(s): {', '.join(missing_cols)}."
            )
            st.info(f"Default {library_type.lower()} library will be used instead.")
            return False
        return True

    # --- Load Warhead Library ---
    if use_custom_libs and warhead_file:
        try:
            warhead_df = load_and_process_sdf(warhead_file)
            if not validate_library(warhead_df, "Warhead"):
                warhead_df = load_and_process_sdf("default_warhead.sdf")
        except Exception as e:
            st.error(f"Could not load Warhead SDF: {e}")
            warhead_df = load_and_process_sdf("default_warhead.sdf")
    else:
        warhead_df = load_and_process_sdf("default_warhead.sdf")

    # --- Load E3 Ligand Library ---
    if use_custom_libs and e3_file:
        try:
            e3_df = load_and_process_sdf(e3_file)
            if not validate_library(e3_df, "E3 Ligand"):
                e3_df = load_and_process_sdf("default_E3ligand.sdf")
        except Exception as e:
            st.error(f"Could not load E3 Ligand SDF: {e}")
            e3_df = load_and_process_sdf("default_E3ligand.sdf")
    else:
        e3_df = load_and_process_sdf("default_E3ligand.sdf")

    # --- Clean and remove duplicates ---
    warhead_df = warhead_df.dropna(subset=["Mol"]).drop_duplicates(subset="Smiles", keep="first")
    e3_df = e3_df.dropna(subset=["Mol"]).drop_duplicates(subset="Smiles", keep="first")

    # --- PROTAC Input Section ---
    st.markdown("**Provide your PROTACs (choose one option):**")
    input_mode = st.radio("Input mode", ["Paste text", "Upload file"])
    protac_entries = []
    if input_mode == "Paste text":
        protac_input = st.text_area("Enter PROTAC name followed by a tab/space and PROTAC SMILES (more than one compound can be pasted)")
        if protac_input.strip():
            for line in protac_input.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    smiles = parts[1]
                    protac_entries.append((name, smiles))
                else:
                    invalid_line = line.strip()
                    st.warning(f"Invalid input format, please enter PROTAC name and SMILES separated by a tab/space")
                    error_messages.append((invalid_line, "Invalid input format", "", "Less than 2 fields (name + SMILES)"))

    elif input_mode == "Upload file":
        st.markdown(
            """
            **📂 Upload a PROTAC file (.txt, .csv, or .sdf):**  
            The file must contain two columns:  
            - `Name` → PROTAC identifier  
            - `Smiles` → PROTAC SMILES string  

            Example (CSV or TXT):  
            ```
            Name,Smiles
            PROTAC_1,CCOCC(=O)N1CCC...
            PROTAC_2,CC(C)OC(=O)NCCC...
            ```
            """
        )
        
        uploaded_file = st.file_uploader("Please, upload your file here", type=["txt", "csv", "sdf"])
        if uploaded_file:
            if uploaded_file.name.endswith(".sdf"):
                sdf_df = PandasTools.LoadSDF(uploaded_file)
                if {"Name", "Smiles"}.issubset(sdf_df.columns):
                    for _, row in sdf_df.iterrows():
                        protac_entries.append((row["Name"], row["Smiles"]))
                else:
                    st.error("SDF must contain 'Name' and 'Smiles' fields.")
            else:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")  # auto-detects comma/tab
                if {"Name", "Smiles"}.issubset(df.columns):
                    for _, row in df.iterrows():
                        protac_entries.append((row["Name"], row["Smiles"]))
                else:
                    st.error("File must contain columns 'Name' and 'Smiles'.")


    if st.button("Split PROTACs"):
        # Email validation (if enabled)
        #if not user_email:
        #    st.error("Please enter your email address before starting the process.")
        #    st.stop()
        #elif not is_valid_email(user_email):
        #    st.error("Please enter a valid email address (e.g., name@example.com) before starting the process.")
        #    st.stop()

        st.info(f"Processing {len(protac_entries)} PROTACs.")

        progress_bar = st.progress(0)  # initialize progress bar
        progress_text = st.empty()      # for dynamic percentage text

        error_messages = []  # Create list to collect all errors
        clean_results = []
        all_results = []
        
        protac_df = pd.DataFrame(protac_entries, columns=["Name", "Smiles"])
        protac_df["Mol"] = protac_df["Smiles"].apply(lambda s: Chem.MolFromSmiles(s) if pd.notnull(s) else None)

        total = len(protac_df)
        for i, (_, row) in enumerate(protac_df.iterrows(), start=1):
            name = row["Name"]
            protac_smiles = row["Smiles"]
            m = Chem.MolFromSmiles(protac_smiles, sanitize=False)
            
            if m is None:
                # Log invalid SMILES (parsing failed)
                log_error(name, "Invalid SMILES", protac_smiles,"RDKit could not parse the SMILES string", error_messages)
            else:
                    try:
                        Chem.SanitizeMol(m)
                    except Exception as e:
                        # Collect sanitization error
                        error_messages.append((name, "Sanitization failed", protac_smiles, str(e)))
                        m = None  # mark as invalid
            if m is None:
                # Skip further processing for this PROTAC
                progress_fraction = i / total
                progress_bar.progress(progress_fraction)
                progress_text.text(f"Processing PROTACs: {int(progress_fraction*100)}% done ({i}/{total})")
                continue


            final_df = split_protac(name, protac_smiles, m, warhead_df, e3_df, error_messages=error_messages)
            if not final_df.empty:
                all_results.append((name, final_df))
                for _, row in final_df.iterrows():
                    clean_results.append({
                        "Protac Name": name,
                        "Protac SMILES": row["Protac SMILES"],
                        "Warhead SMILES": row["Warhead SMILES"],
                        "E3 SMILES": row["E3 SMILES"],
                        "Linker SMILES": row["Linker SMILES"]
                    })

            # update progress bar and text
            progress_fraction = i / total
            progress_bar.progress(progress_fraction)
            progress_text.text(f"Processing PROTACs: {int(progress_fraction*100)}% done ({i}/{total})")

        # clear progress when done
        progress_bar.empty()
        progress_text.empty()

        # --- Save errors if any ---
        # Save results in session state so they persist
        st.session_state["clean_results"] = clean_results
        st.session_state["error_messages"] = error_messages
        st.session_state["processing_done"] = True

    # --- Show results if they exist in session state ---
    if st.session_state.get("error_messages"):
        error_df = pd.DataFrame(st.session_state["error_messages"], columns=["Name", "Error-type", "SMILES", "Details"])
        #st.warning(f"⚠️ {len(st.session_state['error_messages'])} errors were encountered during processing.")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # TXT download
        txt_buffer = io.StringIO()
        error_df.to_csv(txt_buffer, index=False, sep="\t")
        st.download_button(
            "📄 Download error log (TXT)",
            data=txt_buffer.getvalue(),
            file_name=f"protac_error_log_{timestamp}.txt",
            mime="text/plain"
        )

    if st.session_state.get("clean_results"):
        st.success("✅ PROTAC splitting completed successfully.")
        clean_df = pd.DataFrame(st.session_state["clean_results"])
        clean_df['Solution'] = clean_df.groupby('Protac Name').cumcount() + 1
        cols = ['Protac Name', 'Solution', 'Protac SMILES', 'Warhead SMILES', 'E3 SMILES', 'Linker SMILES']
        clean_df = clean_df[cols]

        # CSV download
        csv_buffer = io.StringIO()
        clean_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "💾 Download results as CSV",
            csv_buffer.getvalue(),
            file_name="protac_splitting_results.csv",
            mime="text/csv"
        )

        # TXT download
        txt_buffer = io.StringIO()
        clean_df.to_csv(txt_buffer, index=False, sep="\t")
        st.download_button(
            "📄 Download results as TXT",
            txt_buffer.getvalue(),
            file_name="protac_splitting_results.txt",
            mime="text/plain"
        )

    elif st.session_state.get("processing_done") and not st.session_state.get("clean_results") and not st.session_state.get("error_messages"):
        st.info("⚠️ No valid PROTACs were processed.")

        st.markdown("---")  # Separator for clarity

    st.markdown('<div style="text-align: center; font-size: 13px;"><i> Libraries last updated on 01/10/2025 (warhead and E3 ligand collections). </i></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logos.svg", width=350)
    st.markdown("Bellerophon is developed by [MedChemBeyond group](https://www.cassmedchem.unito.it/) from Molecular Biotechnology and Health Sciences Department (University of Turin) in collaboration with [Alvascience](https://www.alvascience.com/). The Service is meant for non-commercial use only. For info, problems or a personalized version contact giulia.apprato@unito.it")
    st.sidebar.markdown("### You may be interested into our PROTAC-related works")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.5c01497" target="_blank">
            <b>Linker methylation as a strategy to enhance PROTAC oral bioavailability</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">J. Med. Chem., 2025</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("linker-methylation.svg")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://pubs.acs.org/doi/10.1021/acs.jmedchem.4c01200" target="_blank">
            <b>IMHB-mediated chameleonicity in drug design</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">J. Med. Chem., 2024</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("IMHB-cham.svg")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://www.tandfonline.com/doi/full/10.1080/17460441.2025.2467195" target="_blank">
            <b>ADME and PK/PD optimization towards oral PROTACs</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">Expert Opin. Drug Discov., 2024</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("DMPK-review.svg")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://pubs.acs.org/doi/10.1021/acsmedchemlett.3c00362" target="_blank">
            <b>DegraderTCM and ternary complex modeling</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">ACS Med. Chem. Letters, 2023</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("degradertcm.svg")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c00823" target="_blank">
            <b>ChamelogK, experimental descriptor of chamaleonicity</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">J. Med. Chem., 2023</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("chamelogk.svg")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://doi.org/10.1016/j.drudis.2024.103917" target="_blank">
            <b>Exploring the chemical space of orally bioavailable PROTACs</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">Drug Discov. Today, 2024</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("orally_bioavailable.svg")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://pubs.acs.org/doi/full/10.1021/acsmedchemlett.3c00231" target="_blank">
            <b>PROTACs screening pipeline weaknesses</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">ACS Med. Chem. Letters, 2023</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("protacs_pipeline.svg")
    st.sidebar.markdown(
    """
    <div style="font-size:16px;">
        <a href="https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.2c00201" target="_blank">
            <b>Designing soluble PROTACs</b>
        </a><br>
        <span style="font-size:13px; font-style:italic;">J. Med. Chem., 2022</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.sidebar.image("protacs_solubility.svg")       

if __name__ == "__main__":
    main()
