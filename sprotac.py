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
warhead_df = load_and_process_sdf('warhead-protacdb-06-05-2024_customized.sdf')
warhead_df = warhead_df.dropna(subset=['Mol'])
e3_df = load_and_process_sdf('e3-ligand-protacdb-06-05-2024.sdf')
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
    #matched_warheads = [] for debugging

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
    #st.image(Draw.MolToImage(linker_mol, size=(500, 500)), caption='linker mol') #debugging option
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
    f_warhead_df = warhead_df[['Compound ID', 'Target', 'Name', 'PubChem', 'ChEMBL']].copy()
    f_e3_df = e3_df[['Compound ID', 'Target', 'Name', 'PubChem', 'ChEMBL']].copy()

    if 'Compound ID' in f_warhead_df.columns:
        selected_columns_warhead = ['Compound ID', 'Target', 'Name', 'PubChem', 'ChEMBL']
        final_df = pd.merge(final_df, f_warhead_df[selected_columns_warhead], 
                                left_on='warhead ID', right_on='Compound ID', 
                                suffixes=('_warhead', '_warhead'), how='left')

    if 'Compound ID' in f_e3_df.columns:
        selected_columns_e3 = ['Compound ID', 'Target', 'Name', 'PubChem', 'ChEMBL']
        final_df = pd.merge(final_df, f_e3_df[selected_columns_e3], 
                                left_on='E3 ID', right_on='Compound ID', 
                                suffixes=('_warhead', '_E3'), how='left')

    final_df.drop(['Compound ID_warhead', 'Compound ID_E3'], axis=1, inplace=True)

    return final_df


# Function to retrieve and display molecular properties from PubChem using the pubchem ID present in the warhead and E3 ligand dataframe
def retrieve_and_display_pubchem_properties(pubchem_id, smiles, title):
    if not pd.isna(pubchem_id):
        try:
            pubchem_id_numeric = float(pubchem_id)
        except ValueError:
            st.error("Error: Invalid PubChem ID. PubChem ID must be a numeric value.")
            return
        if np.isnan(pubchem_id_numeric):
            st.error("Error: PubChem ID cannot be NaN.")
            return
        
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,TPSA,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount/JSON"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            properties = {}
            try:
                properties_data = data['PropertyTable']['Properties'][0]
                properties = {
                    "Molecular Formula": properties_data.get("MolecularFormula"),
                    "Molecular Weight": properties_data.get("MolecularWeight"),
                    "Canonical SMILES": properties_data.get("CanonicalSMILES"),
                    "TPSA": properties_data.get("TPSA"),
                    "Charge": properties_data.get("Charge"),
                    "Hydrogen Bond Donor Count": properties_data.get("HBondDonorCount"),
                    "Hydrogen Bond Acceptor Count": properties_data.get("HBondAcceptorCount"),
                    "Rotatable Bond Count": properties_data.get("RotatableBondCount")
                }
            except (KeyError, IndexError):
                st.error("Error: Unable to retrieve properties from the response.")
                return
            
            st.markdown(f"### {title} molecular properties")
            properties_table = {
                "Molecular Formula": properties.get('Molecular Formula'),
                "Molecular Weight (g/mol)": properties.get('Molecular Weight'),
                "Canonical SMILES": properties.get('Canonical SMILES'),
                "TPSA (Å²)": properties.get('TPSA'),
                "Charge": properties.get('Charge'),
                "Hydrogen Bond Donor Count": properties.get('Hydrogen Bond Donor Count'),
                "Hydrogen Bond Acceptor Count": properties.get('Hydrogen Bond Acceptor Count'),
                "Rotatable Bond Count": properties.get('Rotatable Bond Count')
            } 
            st.write(f"PubChem ID: [{pubchem_id}](https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_id})")
            st.table(properties_table)
            st.markdown('Molecular properties calculated by PubChem.')
        #else:
        #    st.error(f"Error: Unable to fetch data from PubChem for the {title}.")
    else:
        st.error("No PubChem ID retrieved.")
        if isinstance(smiles, str):
            st.write(f"The {title} provided: {smiles}")
            # Check if the provided SMILES string is valid
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Kekulize the molecule
                st.image(Draw.MolToImage(mol))
            else:
                st.error("Invalid SMILES string provided. Please provide a valid SMILES.")

        elif smiles is not None and not smiles.empty:
            st.write("Here you can find a list of the most similar compounds present in PubChem:")
            perform_similarity_search(smiles)
        else:
            st.write("No DataFrame provided or DataFrame is empty.")
    
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

# Streamlit app------------------------------------------------------------------------------------------------------
def main():
    st.markdown("""
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@900&display=swap" rel="stylesheet">
        </head>
            <div style="text-align: center; font-family: 'Nunito', sans-serif; font-size: 32px;">
            PROTAC splitting tool
        </div>
        """, unsafe_allow_html=True)
    st.image("arv-110-2.svg", use_column_width=True) # figure of ARV-110 building blocks highlighted followed by the caption
    st.write("")
    st.markdown('<div style="text-align: justify"><b>PROTACs</b>, PROteolysis TARgeting Chimeras are heterobifunctional molecules capable of recruiting the ubiquitination complex and to cause the <b>degradation of the target protein</b>. PROTACs are made of three components, a <b>warhead</b> that binds the target, an <b>E3 ligand</b> that recruits the E3 ligase -part of the ubiquitination complex- and a <b>linker</b> joining these moieties. This tool allows to split PROTACs into their components. Warheads (currently 362 unique ligands) and e3 ligand (currently 75 unique ligands) are retrieved from <a href="http://cadd.zju.edu.cn/protacdb/">PROTAC-DB</a>. We hope this will help you investigating PROTACs building block properties and combining them for new design ideas.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Enter your PROTAC SMILES below:**")
    protac_smiles = st.text_input("PROTAC SMILES") # smiles as input

    if st.button("Split PROTAC"):
        if protac_smiles:
            m = Chem.MolFromSmiles(protac_smiles, sanitize=False)
            if m is None:
                st.error('Invalid SMILES, please check for typos')
                return
            else:
                try:
                    Chem.SanitizeMol(m)
                except:
                    st.error('Invalid chemistry, please provide a correct SMILES')
                    return
            
            final_df = split_protac(protac_smiles, warhead_df, e3_df) # applying the function for protac splitting and check (line 42-223)
            st.subheader("Splitting Results:")
            
            columns_to_display = ['Protac SMILES', 'Warhead SMILES', 'E3 SMILES', 'Linker SMILES']
            st.write(final_df[columns_to_display])
            #st.write(final_df) # debugging 
            IPythonConsole.ipython_useSVG = True  # Change output to SVG
            
            if final_df.empty:
                st.write("No matches found for the provided PROTAC SMILES.")
            else:
                for idx, row in final_df.iterrows():
                        st.markdown(f"**Protac 2D structure**")
                        st.image(Draw.MolToImage(row['Protac Mol'], size=(500, 500)), use_column_width=False, output_format='PNG')
                        col1, col2, col3 = st.columns(3)
                        col1.image(Draw.MolToImage(row['Warhead Mol'], size=(250, 300)), caption="Warhead", use_column_width=False)
                        col2.image(Draw.MolToImage(row['Final Linker Mol'], size=(250, 300)), caption="Linker", use_column_width=False)
                        col3.image(Draw.MolToImage(row['E3 Mol'], size=(250, 300)), caption="E3 ligand", use_column_width=False)
                        st.markdown("---") 
                        #st.write(final_df) debugging function only
                # warhead: adding links to PubChem page 
                if row.get('PubChem_warhead') is not None:
                    pubchem_id = row['PubChem_warhead']
                    smiles = final_df['Warhead SMILES'].iloc[0]
                    title = "Warhead"
                retrieve_and_display_pubchem_properties(pubchem_id, smiles, title)

                # e3 ligand: adding links to PubChem page 
                if row.get('PubChem_E3') is not None:
                    pubchem_id = row['PubChem_E3']
                    smiles = final_df['E3 SMILES'].iloc[0]
                    title = "E3 ligand"
                retrieve_and_display_pubchem_properties(pubchem_id, smiles, title)
    
    st.markdown("---")

    st.markdown('<div style="text-align: center; font-size: 13px;"> Last updated from PROTAC-DB on 06/05/2024. </div>', unsafe_allow_html=True)
    st.image("logo.svg", width=200)
    st.image("alvascience_logo.png", width=200)
    st.markdown("Splitting PROTAC is developed by CASSMedChem group from University of Turin in collaboration with [Alvascience](https://www.alvascience.com/). The Service is meant for non-commercial use only. For info, problems or a personalized version contact giulia.apprato@unito.it")
    st.sidebar.markdown("### You may be interested into our PROTAC-related works")
    st.sidebar.markdown("[1. DegraderTCM, ternary complex modeling and PROTACs ranking](https://pubs.acs.org/doi/10.1021/acsmedchemlett.3c00362)")
    st.sidebar.image("degradertcm.svg", use_column_width=True)
    st.sidebar.markdown("[2. ChamelogK, experimental descriptor of chamaleonicity](https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c00823)")
    st.sidebar.image("chamelogk.svg", use_column_width=True)
    st.sidebar.markdown("[3. Orally bioavailable PROTACs chemical space](https://www.sciencedirect.com/science/article/pii/S1359644624000424?via%3Dihub)")
    st.sidebar.image("orally_bioavailable.svg", use_column_width=True)
    st.sidebar.markdown("[4. PROTACs screening pipeline weaknesses](https://pubs.acs.org/doi/full/10.1021/acsmedchemlett.3c00231)")
    st.sidebar.image("protacs_pipeline.svg", use_column_width=True)
    st.sidebar.markdown("[5. Designing soluble PROTACs](https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.2c00201)")
    st.sidebar.image("protacs_solubility.svg", use_column_width=True)       

if __name__ == "__main__":
    main()
