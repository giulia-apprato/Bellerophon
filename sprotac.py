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

#--------------------- Debugging function ---------------------
def split_protac_debug(protac_name, protac_mol, warhead_df, e3_df):
    """Debug version with many checkpoints (writes to Streamlit UI and terminal)."""
    import traceback
    # quick helper to log both app and terminal
    def log(msg, level="info"):
        # use different display styles if desired
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)
        print(msg)

    try:
        log(f"[CP0] Entered split_protac for: {protac_name}")
        if protac_mol is None:
            log(f"[CP0.1] protac_mol is None for {protac_name} — aborting", "error")
            return pd.DataFrame()

        log("[CP1] protac_mol OK — starting warhead scan")

        # Prepare containers
        results = []
        results_df1 = []
        protac_matches = []
        warhead_matched = False

        # quick counters for library sizes
        try:
            n_warheads = len(warhead_df)
            n_e3 = len(e3_df)
        except Exception:
            n_warheads = None
            n_e3 = None
        log(f"[CP1.1] warhead_df length: {n_warheads}, e3_df length: {n_e3}")

        # --- WARHEAD MATCHING ---
        warhead_loop_idx = 0
        for _, warhead_row in warhead_df.iterrows():
            warhead_loop_idx += 1
            if warhead_loop_idx % 50 == 0:
                log(f"[CP1.2] warhead loop iteration: {warhead_loop_idx}")

            # validate warhead entry
            if 'Mol' not in warhead_row or warhead_row['Mol'] is None:
                # skip invalid entries quietly but log occasionally
                if warhead_loop_idx % 100 == 0:
                    log(f"[CP1.3] skipping invalid warhead row at idx {warhead_loop_idx}", "warning")
                continue

            warhead_mol = warhead_row['Mol']
            if not isinstance(warhead_mol, Chem.Mol):
                log(f"[CP1.4] non-Mol warhead at idx {warhead_loop_idx}", "warning")
                continue

            # try substructure matching with try/except
            try:
                matches = protac_mol.GetSubstructMatches(warhead_mol)
            except Exception as e:
                log(f"[CP1.5] Exception during GetSubstructMatches for warhead idx {warhead_loop_idx}: {e}", "warning")
                print(traceback.format_exc())
                continue

            if matches:
                log(f"[CP1.6] Found matches for warhead idx {warhead_loop_idx} (n_matches={len(matches)})")
                w_name = warhead_row.get('Compound ID', f"warhead_idx_{warhead_loop_idx}")
                for match in matches:
                    try:
                        linker_after_w = AllChem.DeleteSubstructs(protac_mol, warhead_mol)
                    except Exception as e:
                        log(f"[CP1.7] DeleteSubstructs(protac, warhead) failed: {e}", "warning")
                        linker_after_w = None
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

        log(f"[CP2] Finished warhead loop — warhead_matched={warhead_matched}")

        if not warhead_matched:
            log(f"[CP2.1] No warhead matched for {protac_name}. Skipping PROTAC.", "warning")
            return pd.DataFrame()

        # append protac_matches to results
        results.extend(protac_matches)
        log(f"[CP2.2] protac_matches saved (count={len(protac_matches)})")

        # --- E3 MATCHING ---
        e3_loop_idx = 0
        e3_matched = False
        for _, e3_row in e3_df.iterrows():
            e3_loop_idx += 1
            if e3_loop_idx % 50 == 0:
                log(f"[CP3] e3 loop iteration: {e3_loop_idx}")

            if 'Mol' not in e3_row or e3_row['Mol'] is None:
                if e3_loop_idx % 100 == 0:
                    log(f"[CP3.1] skipping invalid e3 row at idx {e3_loop_idx}", "warning")
                continue

            e3_mol = e3_row['Mol']
            if not isinstance(e3_mol, Chem.Mol):
                log(f"[CP3.2] non-Mol e3 at idx {e3_loop_idx}", "warning")
                continue

            try:
                matches = protac_mol.GetSubstructMatches(e3_mol)
            except Exception as e:
                log(f"[CP3.3] Exception during GetSubstructMatches for e3 idx {e3_loop_idx}: {e}", "warning")
                print(traceback.format_exc())
                continue

            if matches:
                log(f"[CP3.4] Found matches for E3 idx {e3_loop_idx} (n_matches={len(matches)})")
                e3_name = e3_row.get('Compound ID', f"e3_idx_{e3_loop_idx}")
                for match in matches:
                    try:
                        linker_after_e3 = AllChem.DeleteSubstructs(protac_mol, e3_mol)
                    except Exception as e:
                        log(f"[CP3.5] DeleteSubstructs(protac, e3) failed: {e}", "warning")
                        linker_after_e3 = None
                    results_df1.append({
                        'Protac Mol': protac_mol,
                        'E3 Mol': e3_mol,
                        'Linker Mol': linker_after_e3,
                        'E3 ID': e3_name,
                        'E3 SMILES': mol_to_smiles(e3_mol)
                    })
                e3_matched = True

        log(f"[CP4] Finished E3 loop — e3_matched={e3_matched} (results_df1 count={len(results_df1)})")

        if not e3_matched:
            log(f"[CP4.1] No E3 matched for {protac_name}. Skipping PROTAC.", "warning")
            return pd.DataFrame()

        # --- Build DataFrames and merge ---
        try:
            results_df = pd.DataFrame(results)
            results_df1 = pd.DataFrame(results_df1)
            log(f"[CP5] results_df rows: {len(results_df)}, results_df1 rows: {len(results_df1)}")
        except Exception as e:
            log(f"[CP5.1] Error building interim DataFrames: {e}", "error")
            print(traceback.format_exc())
            return pd.DataFrame()

        if results_df.empty or results_df1.empty:
            log("[CP5.2] One of interim DataFrames is empty — aborting", "warning")
            return pd.DataFrame()

        try:
            final_df = pd.merge(results_df, results_df1, on='Protac Mol', how='inner')
            log(f"[CP6] After merge, final_df rows: {len(final_df)}")
        except Exception as e:
            log(f"[CP6.1] Merge failed: {e}", "error")
            print(traceback.format_exc())
            return pd.DataFrame()

        if final_df.empty:
            log("[CP6.2] final_df empty after merge — aborting", "warning")
            return pd.DataFrame()

        # compute final linkers etc. (wrap in try to catch unexpected errors)
        try:
            final_df['Final Linker Mol'] = final_df.apply(
                lambda row: (AllChem.DeleteSubstructs(row['Linker Mol_x'], row['E3 Mol'])
                             if row['E3 Mol'] is not None and row['Linker Mol_x'] is not None and
                                row['Linker Mol_x'].HasSubstructMatch(row['E3 Mol'])
                             else None),
                axis=1
            )
            log("[CP7] Computed Final Linker Mol")
        except Exception as e:
            log(f"[CP7.1] Error computing Final Linker Mol: {e}", "error")
            print(traceback.format_exc())
            return pd.DataFrame()

        # (continue as in your validated function)...
        # For brevity, perform the remaining validation steps but include a final checkpoint
        try:
            final_df['Linker Mol Check'] = final_df.apply(
                lambda row: (AllChem.DeleteSubstructs(row['Linker Mol_y'], row['Warhead Mol'])
                             if row['Warhead Mol'] is not None and row['Linker Mol_y'] is not None and
                                row['Linker Mol_y'].HasSubstructMatch(row['Warhead Mol'])
                             else None),
                axis=1
            )
            final_df['Linker_check'] = final_df.apply(
                lambda row: True if row['Final Linker Mol'] is not None and row['Linker Mol Check'] is not None and
                mol_to_smiles(row['Final Linker Mol']) == mol_to_smiles(row['Linker Mol Check'])
                else False,
                axis=1
            )
            final_df = final_df[final_df['Linker_check'] == True].copy()
            log(f"[CP8] Linker consistency checked — remaining rows: {len(final_df)}")
        except Exception as e:
            log(f"[CP8.1] Error during linker checks: {e}", "error")
            print(traceback.format_exc())
            return pd.DataFrame()

        # final housekeeping (fragments, heavy atoms, ring count) with checkpoints
        final_df['Linker SMILES'] = final_df['Final Linker Mol'].apply(mol_to_smiles)
        final_df['Final Linker Mol'] = final_df['Linker SMILES'].apply(mol_from_smiles)
        final_df['Num_Disconnected_Fragments'] = final_df['Final Linker Mol'].apply(
            lambda m: len(Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)) if m is not None else 0
        )
        final_df = final_df[final_df['Num_Disconnected_Fragments'] == 1].copy()
        log(f"[CP9] Fragment filtering done — rows now: {len(final_df)}")

        # heavy atom and ring checks
        final_df['Protac heavy atoms'] = final_df['Protac Mol'].apply(lambda m: rdMolDescriptors.CalcNumHeavyAtoms(m) if m is not None else None)
        final_df['Warhead heavy atoms'] = final_df['Warhead Mol'].apply(lambda m: rdMolDescriptors.CalcNumHeavyAtoms(m) if m is not None else None)
        final_df['E3 heavy atoms'] = final_df['E3 Mol'].apply(lambda m: rdMolDescriptors.CalcNumHeavyAtoms(m) if m is not None else None)
        final_df['Linker heavy atoms'] = final_df['Final Linker Mol'].apply(lambda m: rdMolDescriptors.CalcNumHeavyAtoms(m) if m is not None else None)
        final_df['Sum of heavy atoms'] = final_df[['Warhead heavy atoms', 'E3 heavy atoms', 'Linker heavy atoms']].sum(axis=1)
        final_df['Heavy atom check'] = final_df.apply(
            lambda r: (r['Sum of heavy atoms'] is not None and r['Protac heavy atoms'] is not None and r['Sum of heavy atoms'] == r['Protac heavy atoms']),
            axis=1
        )
        final_df = final_df[final_df['Heavy atom check'] == True].copy()
        log(f"[CP10] Heavy atom check passed — rows: {len(final_df)}")

        # ring counts
        final_df['Protac ring count'] = final_df['Protac Mol'].apply(lambda m: rdMolDescriptors.CalcNumRings(m) if m is not None else None)
        final_df['Warhead ring count'] = final_df['Warhead Mol'].apply(lambda m: rdMolDescriptors.CalcNumRings(m) if m is not None else None)
        final_df['E3 ring count'] = final_df['E3 Mol'].apply(lambda m: rdMolDescriptors.CalcNumRings(m) if m is not None else None)
        final_df['Linker ring count'] = final_df['Final Linker Mol'].apply(lambda m: rdMolDescriptors.CalcNumRings(m) if m is not None else None)
        final_df['Sum of ring count'] = final_df[['Warhead ring count', 'E3 ring count', 'Linker ring count']].sum(axis=1)
        final_df['Ring count check'] = final_df.apply(
            lambda r: (r['Sum of ring count'] is not None and r['Protac ring count'] is not None and r['Sum of ring count'] == r['Protac ring count']),
            axis=1
        )
        final_df = final_df[final_df['Ring count check'] == True].copy()
        log(f"[CP11] Ring count check passed — rows: {len(final_df)}")

        # final dedup
        final_df['Warhead SMILES'] = final_df['Warhead Mol'].apply(mol_to_smiles)
        final_df['E3 SMILES'] = final_df['E3 Mol'].apply(mol_to_smiles)
        final_df = final_df.drop_duplicates(subset=['Warhead SMILES', 'Linker SMILES', 'E3 SMILES'], keep='first').reset_index(drop=True)
        log(f"[CP12] Dedup done — final rows: {len(final_df)}")

        # keep only useful columns
        cols_keep = ['Protac Mol', 'Protac SMILES', 'Warhead Mol', 'Warhead SMILES', 'E3 Mol', 'E3 SMILES',
                     'Final Linker Mol', 'Linker SMILES', 'warhead ID', 'E3 ID']
        final_df = final_df[[c for c in cols_keep if c in final_df.columns]].copy()

        log(f"[CP13] Completed splitting for {protac_name} — returning {len(final_df)} solutions")
        return final_df

    except Exception as e:
        st.error(f"Unhandled exception in split_protac_debug for {protac_name}: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()
#--------------------- End Debugging function ---------------------

# Define the splitting function
def split_protac(protac_name, protac_mol, warhead_df, e3_df, batch_mode=False):
    """
    Splits a single PROTAC (given as RDKit mol) into warhead, linker and E3 ligand candidates
    using the provided warhead/e3 libraries. Returns a DataFrame with valid solutions (may be empty).
    """
    # Safety: if protac mol not provided, return empty DF
    if protac_mol is None:
        if not batch_mode:
            st.error("❌❌❌ Could not parse PROTAC SMILES ❌❌❌")
            return pd.DataFrame()
    #st.info("🔍 Starting splitting process...")

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
            if not batch_mode:
                st.warning(f"⚠️ Substructure matching failed for {protac_name} (warhead): {e}")
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
        if not batch_mode:
            st.info(f"ℹ️ No warhead match found in library for {protac_name}. Skipping this PROTAC.")
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
            if not batch_mode:
                st.warning(f"⚠️ Substructure matching failed for {protac_name} (E3): {e}")
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
    final_df = final_df[final_df['Heavy atom check'] == True].copy()

    if final_df.empty:
        if not batch_mode:
            st.warning(f"⚠️ No combinations passed the heavy atom check for {protac_name}." 
                    "This often happens if the substructure decomposition isn't perfect."
                    "Please check if the warhead/E3 ligand are present in the libraries")
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
    final_df = final_df[final_df['Ring count check'] == True].copy()

    if final_df.empty:
        if not batch_mode:
            st.warning(f"⚠️ No combinations passed the ring count check for {protac_name}." 
                    "This often happens if the substructure decomposition isn't perfect."
                    "Please check if the warhead/E3 ligand are present in the libraries")
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

    # --- Display images if not in batch mode ---
    if not batch_mode and not final_df.empty:
        st.subheader(f"Results for {protac_name}")
        st.write(final_df[['Protac SMILES', 'Warhead SMILES', 'E3 SMILES', 'Linker SMILES']])
        for i, (_, row) in enumerate(final_df.iterrows(), start=1):
            st.markdown(f"### Solution {i}")
            st.image(Draw.MolToImage(row['Protac Mol'], size=(500, 500)), output_format='PNG')
            col1, col2, col3 = st.columns(3)
            col1.image(Draw.MolToImage(row['Warhead Mol'], size=(250, 300)), caption="Warhead")
            col2.image(Draw.MolToImage(row['Final Linker Mol'], size=(250, 300)), caption="Linker")
            col3.image(Draw.MolToImage(row['E3 Mol'], size=(250, 300)), caption="E3 ligand")
            st.markdown("---")

    return final_df

#Streamlit app--------------------------------------------------------------
import io
import streamlit.components.v1 as components
def main():
    st.image("bellerophon_GA.svg") 
    st.write("")
    st.markdown(
    '<div style="text-align: justify">'
    'Bellerophon is a computational tool designed to automatically split PROTACs into their three components: warhead, linker, and E3 ligase ligand.<br><br>'
    'It identifies the warhead and E3 ligand from a curated library, either the default one provided by the authors or a user-uploaded version. Users can input PROTACs by pasting the name and SMILES or uploading a file.<br><br>'
    'Bellerophon supports data curation and rational design by enabling the comparison and recombination of validated building blocks for efficient PROTAC discovery.'
    '</div>',
    unsafe_allow_html=True
)
    st.write("")

    # Upload datasets (warheads/E3)
    st.markdown("**Upload your libraries (default ones will be used otherwise):**")
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
        protac_input = st.text_area("Enter PROTAC name followed by a tab/space and PROTAC SMILES (more than one compound can be pasted)")
        if protac_input.strip():
            for line in protac_input.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    smiles = parts[1]
                    protac_entries.append((name, smiles))
                else:
                    st.warning(f"Warning, skipping invalid compound: {name}")

    elif input_mode == "Upload file":
        uploaded_file = st.file_uploader("Upload TXT/CSV file with PROTAC name followed by a tab/space and PROTAC SMILES (one compound per line)", type=["txt", "csv", "sdf"])
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
        if not protac_entries:
            st.error("No valid PROTACs provided.")
            return

        BATCH_THRESHOLD = 20 # Number of PROTACs above which batch mode is ACTIVATED
        batch_mode = len(protac_entries) > BATCH_THRESHOLD

        if batch_mode:
            st.info(f"Processing {len(protac_entries)} PROTACs in batch mode, per-PROTAC warnings and images are suppressed.")

        progress_bar = st.progress(0)  # initialize progress bar
        progress_text = st.empty()      # for dynamic percentage text

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
                if not batch_mode:
                    st.warning(f"Warning, invalid SMILES skipped: {name} ({protac_smiles})")
                continue
            try:
                Chem.SanitizeMol(m)
            except:
                if not batch_mode:
                    st.warning(f"Warning, invalid chemistry skipped: {name} ({protac_smiles})")
                continue

            final_df = split_protac(name, m, warhead_df, e3_df, batch_mode=batch_mode)
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

        if clean_results:
            # Add a 'Solution' column corresponding to each solution number per PROTAC
            clean_df = pd.DataFrame(clean_results)
            clean_df['Solution'] = clean_df.groupby('Protac Name').cumcount() + 1

            # Reorder columns for clarity
            cols = ['Protac Name', 'Solution', 'Protac SMILES', 'Warhead SMILES', 'E3 SMILES', 'Linker SMILES']
            clean_df = clean_df[cols]

            # Download buttons before showing results
            csv_buffer = io.StringIO()
            clean_df.to_csv(csv_buffer, index=False)
            st.download_button("💾 Download results as CSV", csv_buffer.getvalue(),
                            file_name="protac_splitting_results.csv", mime="text/csv")

            txt_buffer = io.StringIO()
            clean_df.to_csv(txt_buffer, index=False, sep="\t")
            st.download_button("📄 Download results as TXT", txt_buffer.getvalue(),
                            file_name="protac_splitting_results.txt", mime="text/plain")

            st.markdown("---")

    st.markdown('<div style="text-align: center; font-size: 13px;"><i> Libraries last updated on 01/10/2025 (warhead and E3 ligand collections). </i></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logos.svg", width=350)
    st.markdown("Bellerophon is developed by MedChemBeyond group from Molecular Biotechnology and Health Sciences Department (University of Turin) in collaboration with [Alvascience](https://www.alvascience.com/). The Service is meant for non-commercial use only. For info, problems or a personalized version contact giulia.apprato@unito.it")
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
