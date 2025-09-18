import streamlit as st
import pandas as pd
import rdkit
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from IPython.display import HTML
import requests
IPythonConsole.ipython_useSVG = True
PandasTools.RenderImagesInAllDataFrames(images=True)


# Load SDF files into DataFrames and select relevant columns
def load_and_process_sdf(filename):
    df = PandasTools.LoadSDF(filename)
    df['Mol'] = df['Smiles'].apply(Chem.MolFromSmiles)
    return df

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

# Load and process the SDF files
warhead_df = load_and_process_sdf('default_warhead.sdf')
warhead_df = warhead_df.dropna(subset=['Mol'])
e3_df = load_and_process_sdf('default_E3ligand.sdf')
e3_df = e3_df.dropna(subset=['Mol'])

# warhead and e3 ligand dataset curation
warhead_df['Smiles'] = warhead_df['Smiles'].apply(lambda smiles: rdMolStandardize.StandardizeSmiles(smiles) if smiles else None)
warhead_df['Mol'] = warhead_df['Smiles'].apply(mol_from_smiles)
warhead_df = warhead_df.drop_duplicates(subset='Smiles', keep='first') # 365 unique warheads
e3_df['Smiles'] = e3_df['Smiles'].apply(lambda smiles: rdMolStandardize.StandardizeSmiles(smiles) if smiles else None)
e3_df['Mol'] = e3_df['Smiles'].apply(mol_from_smiles)
e3_df = e3_df.drop_duplicates(subset='Smiles', keep='first') # 82 unique e3 ligands

# Define the splitting function
def split_protac(protac_smiles, warhead_df, e3_df):
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    
    if protac_mol is None:
        return None, None, "Invalid PROTAC SMILES. No mol object was created."

    results = []
    results_df1 = []
    #matched_warheads = [] #debugging option

    for _, warhead_row in warhead_df.iterrows():
        warhead_mol = warhead_row['Mol']
        matches = protac_mol.GetSubstructMatches(warhead_mol)
        w_name = warhead_row['Compound ID']
        #st.image(Draw.MolToImage(warhead_mol, size=(500, 500))) #debugging option
        #st.write('Warhead compound ID: ', w_name) #debugging option
        
        # Check if warhead_mol is None (debug)        
        if warhead_mol is None:
            st.write("Warhead molecule for Compound ID is none, please check the smiles or the list of warhead provided:", w_name)
            break
    
        # Check if the SMILES string is valid (debug)
        warhead_smiles = Chem.MolToSmiles(warhead_mol)
        if warhead_smiles is None:
            st.write("Failed to convert warhead molecule to SMILES for Compound ID:", w_name)
            break

        # Append all matched warheads to the list (debug)
        #for match in matches:
        #    matched_warheads.append(warhead_mol)

    #for idx, warhead_mol in enumerate(matched_warheads):
    #    st.image(Draw.MolToImage(warhead_mol, size=(500, 500)), caption=f'Matched Warhead Molecule {idx+1}')

        for match in matches:
            linker_mol = AllChem.DeleteSubstructs(protac_mol, warhead_mol)
            results.append({
                'Protac Mol': protac_mol,
                'Warhead Mol': warhead_mol,
                'Linker Mol': linker_mol,
                'warhead ID': w_name,
                'Protac SMILES': protac_smiles,
                'Warhead SMILES': Chem.MolToSmiles(warhead_mol)
            })
            #st.image(Draw.MolToImage(linker_mol, size=(500, 500))) #debugging option

    for _, e3_row in e3_df.iterrows():
        e3_mol = e3_row['Mol']
        matches = protac_mol.GetSubstructMatches(e3_mol)
        e3_name = e3_row['Compound ID']
        #st.image(Draw.MolToImage(e3_mol, size=(500, 500))) #debugging option
        #st.write('E3 ligand compound ID: ', e3_name) #debugging option

        if matches:
            linker_mol = AllChem.DeleteSubstructs(protac_mol, e3_mol)
            results_df1.append({
                'Protac Mol': protac_mol,
                'E3 Mol': e3_mol,
                'Linker Mol': linker_mol,
                'E3 ID': e3_name,
                'E3 SMILES': Chem.MolToSmiles(e3_mol)
            })

    #st.image(Draw.MolToImage(protac_mol, size=(500, 500)), caption='protac mol') #debugging option
    #st.image(Draw.MolToImage(warhead_mol, size=(500, 500)), caption='warhead mol') #debugging option
    #if 'linker_mol' in locals() and linker_mol is not None: #debugging option
    #    st.image(Draw.MolToImage(linker_mol, size=(500, 500)), caption='linker mol') #debugging option
    #else: #debugging option
    #    st.warning("No linker molecule was generated for this PROTAC") #debugging option
    #st.write('protac smiles: ', protac_smiles) #debugging option
    #st.write('warhead smiles: ', Chem.MolToSmiles(warhead_mol)) #debugging option
    #st.write('Number of warhead found: ', len(w_name)) #debugging option
    #st.write('Number of e3 ligand found: ', len(e3_name)) #debugging option
    #st.write("Length of the first dataframe:", len(results)) #debugging option
    #st.write('First dataframe containing protac mol, warhead mol, linker mol, warhead ID, protac smiles, warhead smiles: ', results) #debugging option
    #st.write(results) #debugging option
    #st.write("Length of second dataframe:", len(results_df1)) #debugging option
    #st.write('Second dataframe containing protac mol, e3 mol, linker mol, e3 ID, protac smiles, e3 smiles: ', results) #debugging
    #st.write(results_df1) #debugging option
    
    
    final_df = pd.merge(pd.DataFrame(results), pd.DataFrame(results_df1), on='Protac Mol')

    final_df['Final Linker Mol'] = final_df.apply(
        lambda row: AllChem.DeleteSubstructs(row['Linker Mol_x'], row['E3 Mol']) 
                    if row['E3 Mol'] and row['Linker Mol_x'].HasSubstructMatch(row['E3 Mol'])
                    else None, 
        axis=1)

    final_df['Linker Mol Check'] = final_df.apply(
        lambda row: AllChem.DeleteSubstructs(row['Linker Mol_y'], row['Warhead Mol']) 
                        if row['Warhead Mol'] and row['Linker Mol_y'].HasSubstructMatch(row['Warhead Mol'])
                        else None, 
        axis=1)

    final_df['Linker_check'] = final_df.apply(
        lambda row: row['Final Linker Mol'] and row['Linker Mol Check'] and 
        Chem.MolToSmiles(row['Final Linker Mol']) == Chem.MolToSmiles(row['Linker Mol Check']),
        axis=1)
    
    final_df = final_df.drop(columns=['Linker Mol_x', 'Linker Mol_y', 'Linker Mol Check'])
    final_df = final_df[final_df['Linker_check'] == True]
    final_df['Linker SMILES'] = final_df['Final Linker Mol'].apply(mol_to_smiles)
    
    # Function to count the number of disconnected fragments for a molecule
    def count_disconnected_fragments(mol):
        if mol:
            # Get the disconnected fragments for the molecule
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            return len(fragments)
        return 0

    # Count the number of disconnected fragments for each molecule in "Final Linker Mol"
    final_df['Num_Disconnected_Fragments'] = final_df['Final Linker Mol'].apply(count_disconnected_fragments)
    final_df = final_df[final_df['Num_Disconnected_Fragments'] == 1]
    
    # Function to calculate the number of heavy atoms for a given RDKit molecule
    def cal_num_hatoms(mol):
        if mol:
            return rdMolDescriptors.CalcNumHeavyAtoms(mol)
        else:
            return None
    # Applying the function to each building block of the protac    
    final_df['Protac heavy atoms'] = final_df['Protac Mol'].apply(cal_num_hatoms)
    final_df['Warhead heavy atoms'] = final_df['Warhead Mol'].apply(cal_num_hatoms)
    final_df['E3 heavy atoms'] = final_df['E3 Mol'].apply(cal_num_hatoms)
    final_df['Linker heavy atoms'] = final_df['Final Linker Mol'].apply(cal_num_hatoms)

    # Add new column for the sum of warhead, linker and E3 ligand heavy atoms
    final_df['Sum of heavy atoms'] = final_df[['Warhead heavy atoms', 'E3 heavy atoms', 'Linker heavy atoms']].sum(axis=1)

    # Checking if the sum of heavy atoms is equal to protac heavy atoms
    final_df['Heavy atom check'] = final_df.apply(
        lambda row: True if row['Sum of heavy atoms'] is not None and row['Protac heavy atoms'] is not None and 
        row['Sum of heavy atoms'] == row['Protac heavy atoms']
        else False, axis=1)
    final_df = final_df[final_df['Heavy atom check'] == True]
    
    # Function to calculate the number of rings for a given RDKit molecule
    def ring_count(mol):
        if mol:
            return rdMolDescriptors.CalcNumRings(mol)
        else:
            return None

    # I have to convert mol to smiles and then to mol again otherwise I get an error when trying to calculate the ring count for the linker
    final_df['Linker SMILES'] =  final_df['Final Linker Mol'].apply(mol_to_smiles) 
    final_df['Final Linker Mol'] =  final_df['Linker SMILES'].apply(mol_from_smiles) 

    # Applying the ring count function to each protac building block
    final_df['Protac ring count'] = final_df['Protac Mol'].apply(ring_count)
    final_df['Warhead ring count'] = final_df['Warhead Mol'].apply(ring_count)
    final_df['E3 ring count'] = final_df['E3 Mol'].apply(ring_count)
    final_df['Linker ring count'] = final_df['Final Linker Mol'].apply(ring_count)

    # Add new column for the sum of warhead, linker and E3 ligand ring count
    final_df['Sum of ring count'] = final_df[['Warhead ring count', 'E3 ring count', 'Linker ring count']].sum(axis=1)

    # Checking if the sum of ring count of the building blocks is equal to protac ring count
    final_df['Ring count check'] = final_df.apply(
        lambda row: True if row['Sum of ring count'] is not None and row['Protac ring count'] is not None and 
        row['Sum of ring count'] == row['Protac ring count']
        else False, axis=1)
    final_df = final_df[final_df['Ring count check'] == True]
    
    #Adding chembl and pubchem info for warhead and E3 ligand from the intial dataset warhead_df and e3_df
    f_warhead_df = warhead_df[['Compound ID']].copy()
    f_e3_df = e3_df[['Compound ID']].copy()

    if 'Compound ID' in f_warhead_df.columns:
        selected_columns_warhead = ['Compound ID']
        final_df = pd.merge(final_df, f_warhead_df[selected_columns_warhead], 
                                left_on='warhead ID', right_on='Compound ID', 
                                suffixes=('_warhead', '_warhead'), how='left')

    if 'Compound ID' in f_e3_df.columns:
        selected_columns_e3 = ['Compound ID']
        final_df = pd.merge(final_df, f_e3_df[selected_columns_e3], 
                                left_on='E3 ID', right_on='Compound ID', 
                                suffixes=('_warhead', '_E3'), how='left')

    final_df.drop(['Compound ID_warhead', 'Compound ID_E3'], axis=1, inplace=True)

    return final_df
    
# Function to search for similar compounds to the warhead or the E3 ligand if the pubchem ID is not present in the original dataframe    
def perform_similarity_search(query_smiles, threshold=90, max_records=3):
    # Construct the query URL for similarity search
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{query_smiles}/cids/JSON?Threshold={threshold}&MaxRecords={max_records}"

    # Send a GET request to PubChem REST API
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        if 'IdentifierList' in data:
            # Extract similar compound CIDs
            similar_cids = data['IdentifierList']['CID']
            #st.markdown("Similar CIDs:"+ str(similar_cids))
            
            # Optionally, retrieve additional information for each CID
            for cid in similar_cids:
                # Process each similar compound CID (e.g., retrieve additional information)
                process_similar_compound(cid, query_smiles)
        else:
            st.markdown("No similar compounds found.")
    else:
        st.markdown("Error: Failed to retrieve data from PubChem.")
        st.markdown(f"Error: Failed to retrieve data from PubChem. Response status code: {response.status_code}")

def process_similar_compound(cid, query_smiles):
    compound_data = []
    # Retrieve additional information for a similar compound CID
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Process the retrieved data (e.g., extract SMILES)
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        
        # Calculate similarity using Tanimoto coefficient
        mol1 = Chem.MolFromSmiles(query_smiles)
        mol2 = Chem.MolFromSmiles(smiles)
        fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        similarity = AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
        
        # Add the data to the list
        compound_data.append({
            "Compound CID": cid,
            "SMILES": smiles,
            "Similarity": similarity,
        })
        
        compound_df = pd.DataFrame(compound_data)
        st.table(compound_df)
        st.write(f"CID: [{cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
        # Convert SMILES to Mol object
        mol = Chem.MolFromSmiles(smiles)
        
        # Display the Mol object as an image
        if mol:
            st.image(Draw.MolToImage(mol))
        else:
            st.error("Error: Failed to generate Mol object from SMILES.")
    else:
        st.error(f"Error: Failed to retrieve data for compound CID {cid}.")
    
# Convert RDKit Mol objects to images
def mol_to_image(mol, size=(300, 300)):
    return Draw.MolToImage(mol, size=size)

#Streamlit app--------------------------------------------------------------
import io
def main():
    st.image("bellerophon_GA.svg") 
    st.write("")
    st.markdown('<div style="text-align: justify"><b>PROTACs</b>, PROteolysis TArgeting Chimeras, are innovative molecules that degrade disease-related proteins by joining a warhead and an E3 ligase ligand through a linker. Despite their potential, no dedicated tool has existed to easily dissect their structures. <b>Bellerophon</b> fills this gap: a free, intuitive platform that splits PROTACs into their components, streamlining data curation, high-throughput analysis, and rational design.</div>', unsafe_allow_html=True)
    st.write("")

    # Upload datasets (warheads/E3)
    st.markdown("**Upload your datasets (optional):**")
    warhead_file = st.file_uploader("Upload Warhead SDF", type=["sdf"])
    e3_file = st.file_uploader("Upload E3 Ligand SDF", type=["sdf"])

    if warhead_file:
        warhead_df = load_and_process_sdf(warhead_file)
    else:
        warhead_df = load_and_process_sdf('default_warhead.sdf')

    if e3_file:
        e3_df = load_and_process_sdf(e3_file)
    else:
        e3_df = load_and_process_sdf('default_E3ligand.sdf')

    warhead_df = warhead_df.dropna(subset=['Mol']).drop_duplicates(subset='Smiles', keep='first')
    e3_df = e3_df.dropna(subset=['Mol']).drop_duplicates(subset='Smiles', keep='first')

    st.markdown("**Provide your PROTACs (choose one option):**")
    input_mode = st.radio("Input mode", ["Paste text", "Upload file"])

    protac_entries = []
    if input_mode == "Paste text":
        st.markdown("Format: `Name SMILES` (one per line, separated by space or tab).")
        protac_input = st.text_area("Enter PROTAC names + SMILES")
        if protac_input.strip():
            for line in protac_input.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    smiles = parts[1]
                    protac_entries.append((name, smiles))
                else:
                    st.warning(f"Skipping invalid line: {line}")

    elif input_mode == "Upload file":
        uploaded_file = st.file_uploader("Upload TXT/CSV file with Protac Name + SMILES", type=["txt", "csv", "sdf"])
        if uploaded_file:
            if uploaded_file.name.endswith(".sdf"):
                sdf_df = PandasTools.LoadSDF(uploaded_file)
                if {"Protac Name", "Protac SMILES"}.issubset(sdf_df.columns):
                    for _, row in sdf_df.iterrows():
                        protac_entries.append((row["Protac Name"], row["Protac SMILES"]))
                else:
                    st.error("SDF must contain 'Protac Name' and 'Protac SMILES' fields.")
            else:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")  # auto-detects comma/tab
                if {"Protac Name", "Protac SMILES"}.issubset(df.columns):
                    for _, row in df.iterrows():
                        protac_entries.append((row["Protac Name"], row["Protac SMILES"]))
                else:
                    st.error("File must contain columns 'Protac Name' and 'Protac SMILES'.")

    if st.button("Split PROTACs"):
        if not protac_entries:
            st.error("No valid PROTACs provided.")
            return

        clean_results = []
        all_results = []

        for name, protac_smiles in protac_entries:
            m = Chem.MolFromSmiles(protac_smiles, sanitize=False)
            if m is None:
                st.warning(f"Invalid SMILES skipped: {name} ({protac_smiles})")
                continue
            try:
                Chem.SanitizeMol(m)
            except:
                st.warning(f"Invalid chemistry skipped: {name} ({protac_smiles})")
                continue

            final_df = split_protac(protac_smiles, warhead_df, e3_df)
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

        if clean_results:
            clean_df = pd.DataFrame(clean_results)

            # Download buttons before showing results
            csv_buffer = io.StringIO()
            clean_df.to_csv(csv_buffer, index=False)
            st.download_button("💾 Download results as CSV", csv_buffer.getvalue(),
                               file_name="protac_splitting_results.csv", mime="text/csv")

            txt_buffer = io.StringIO()
            clean_df.to_csv(txt_buffer, index=False, sep="\t")
            st.download_button("📄 Download results as TXT", txt_buffer.getvalue(),
                               file_name="protac_splitting_results.txt", mime="text/plain")

            # Detailed per-PROTAC visualization
            for name, final_df in all_results:
                protac_smiles = final_df["Protac SMILES"].iloc[0]
                st.subheader(f"Results for {name}: {protac_smiles}")
                st.write(final_df[['Protac SMILES', 'Warhead SMILES', 'E3 SMILES', 'Linker SMILES']])

                for _, row in final_df.iterrows():
                    st.markdown("**Protac 2D structure**")
                    st.image(Draw.MolToImage(row['Protac Mol'], size=(500, 500)), output_format='PNG')
                    col1, col2, col3 = st.columns(3)
                    col1.image(Draw.MolToImage(row['Warhead Mol'], size=(250, 300)), caption="Warhead")
                    col2.image(Draw.MolToImage(row['Final Linker Mol'], size=(250, 300)), caption="Linker")
                    col3.image(Draw.MolToImage(row['E3 Mol'], size=(250, 300)), caption="E3 ligand")
                    st.markdown("---")


    st.markdown('<div style="text-align: center; font-size: 13px;"> Last updated from PROTAC-DB on 06/05/2024. </div>', unsafe_allow_html=True)
    st.image("logo.svg", width=200)
    st.image("alvascience_logo.png", width=200)
    st.markdown("Splitting PROTAC is developed by CASSMedChem group from University of Turin in collaboration with [Alvascience](https://www.alvascience.com/). The Service is meant for non-commercial use only. For info, problems or a personalized version contact giulia.apprato@unito.it")
    st.sidebar.markdown("### You may be interested into our PROTAC-related works")
    st.sidebar.markdown("[1. DegraderTCM, ternary complex modeling and PROTACs ranking](https://pubs.acs.org/doi/10.1021/acsmedchemlett.3c00362)")
    st.sidebar.image("degradertcm.svg")
    st.sidebar.markdown("[2. ChamelogK, experimental descriptor of chamaleonicity](https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c00823)")
    st.sidebar.image("chamelogk.svg")
    st.sidebar.markdown("[3. Orally bioavailable PROTACs chemical space](https://www.sciencedirect.com/science/article/pii/S1359644624000424?via%3Dihub)")
    st.sidebar.image("orally_bioavailable.svg")
    st.sidebar.markdown("[4. PROTACs screening pipeline weaknesses](https://pubs.acs.org/doi/full/10.1021/acsmedchemlett.3c00231)")
    st.sidebar.image("protacs_pipeline.svg")
    st.sidebar.markdown("[5. Designing soluble PROTACs](https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.2c00201)")
    st.sidebar.image("protacs_solubility.svg")       

if __name__ == "__main__":
    main()
