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