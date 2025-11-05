# **Bellerophon: Automated PROTAC Decomposition Tool**

[![Streamlit App](https://img.shields.io/badge/Try%20Online-Streamlit-brightgreen)](https://bellerophon-protacs-splitting-app.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/Chemistry-RDKit-lightblue.svg)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io)

---

## 🚀 **Overview**

**Bellerophon** is a computational tool that **automatically decomposes heterobifunctional degraders (PROTACs)** into their three structural components:  
**Warhead** (target-binding ligand), **Linker**, and **E3 Ligase Ligand**.

Accurately splitting PROTACs into their constituent parts is a complex task. Existing resources provide valuable manually curated information but lack automated, flexible parsing.  
Bellerophon overcomes this limitation with a **structure-based algorithm** that identifies and extracts these fragments directly from a PROTAC’s molecular structure.

By standardizing the decomposition process, **Bellerophon** enables:
-  High-throughput annotation and data curation  
-  Comparison and recombination of validated building blocks  
-  Structure-driven *de novo* PROTAC design workflows  

---

## 🔍 **How It Works**

Bellerophon uses a **multi-step, structure-based comparison** procedure:

1. The input PROTAC is matched against a curated **warhead library**.  
2. Once the warhead is identified and removed, the remaining structure is compared against the **E3 ligase library**.  
3. The **linker** is inferred from the remaining molecular fragment.  
4. The process is repeated in reverse (starting from the E3 ligand) to ensure **consistency**.  
5. Only decompositions yielding identical linkers in both directions are retained.  

To minimize false positives, the algorithm applies multiple **structural integrity checks**, ensuring the identified components fully reconstruct the original PROTAC (heavy atom count, ring count, and connectivity).  

---

## ⚙️ **Implementation**

- **Language:** Python  
- **Core library:** RDKit  
- **Interface:** Streamlit  
- **Source code and default libraries:** Available in this repository  
- **Web app:** [🌐 Bellerophon Streamlit App](https://bellerophon-protacs-splitting-app.streamlit.app/)  

---

## ✨ **Features**

-  Automatic identification of warhead, linker, and E3 ligand  
-  Consistency validation via heavy-atom and ring-count checks  
-  Built-in curated libraries (warheads and E3 ligases)  
-  Option to upload custom `.sdf` libraries  
-  Batch processing of multiple PROTACs  
-  Downloadable results in `.csv` or `.txt` format  
-  Internal error logging and reporting  

---

## 📂 **Quick Start**

### **Run Online**
👉 Try it instantly on Streamlit:  
[https://bellerophon-protacs-splitting-app.streamlit.app/](https://bellerophon-protacs-splitting-app.streamlit.app/)

### **Run Locally**

```bash
# Clone the repository
git clone https://github.com/yourusername/bellerophon.git
cd bellerophon

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run sprotac.py
```

## 🧾 **Usage**

### **Input Options**

1. **Paste Mode:**  
   Directly paste PROTAC names and SMILES strings.

2. **File Upload:**  
   Upload `.txt`, `.csv`, or `.sdf` files containing the columns **Name** and **SMILES**.

---

### **Output**

The tool returns a table listing all valid decompositions, including:

- **Compound Name**  
- **PROTAC SMILES**  
- **Warhead SMILES**  
- **Linker SMILES**  
- **E3 Ligand SMILES**

All results can be downloaded as `.csv` or `.txt` files.  
A separate file containing any **errors encountered** during processing is also available.

---

## 📚 **Default Libraries**

Curated collections of **warheads** and **E3 ligands** are included in the repository.  
These were compiled from degraders reported in **clinical trials** and **recent literature**.  

Users can also upload **custom `.sdf` libraries** to tailor the decomposition process to specific datasets or design goals.

---

## 👩‍🔬 **Authors**

Developed by the [**MedChemBeyond Group**](https://www.cassmedchem.unito.it/),  
**Department of Molecular Biotechnology and Health Sciences**,  
**University of Turin**, in collaboration with [**Alvascience**](https://www.alvascience.com/).

📧 **Contact:** [giulia.apprato@unito.it](mailto:giulia.apprato@unito.it)

---

## 📄 **License**

This repository is distributed under the **MIT License** (see [`LICENSE`](./LICENSE)).  
Use is permitted for **non-commercial purposes only**.

---

## 🧾 **Citation**

If you use **Bellerophon** in your research, please cite:

> **Bellerophon: A structure-based tool for automatic PROTAC decomposition**  
> Web app: [https://bellerophon-protacs-splitting-app.streamlit.app/](https://bellerophon-protacs-splitting-app.streamlit.app/)

---

## 🌟 **Acknowledgments**

Special thanks to contributors from the **University of Turin** and **Alvascience**  
for supporting the development of **Bellerophon**.

---

