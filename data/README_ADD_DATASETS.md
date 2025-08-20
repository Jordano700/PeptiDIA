# Adding Your Dataset to the Enhanced Peptide Validator

This tool now uses **universal data loading** that works with ANY dataset naming convention! 

## 🚀 Quick Start (3 Steps)

### 1. Create the Folder Structure
```
data/
└── YourDatasetName/           # Use your lab/project name
    ├── short_gradient/        # Fast gradient data
    │   ├── FDR_1/            # Baseline files (1% FDR)
    │   ├── FDR_20/           # Training files (20% FDR)  
    │   └── FDR_50/           # Training files (50% FDR)
    └── long_gradient/         # Slow gradient data
        └── FDR_1/            # Ground truth files (1% FDR)
```

### 2. Add Your Files
Drop your `.parquet` files into the appropriate folders. **Any naming convention works!**

Examples of supported filename patterns:
- `LC60_Method_A_Sample_001_FDR1.parquet`
- `MS120-DIA15_Condition_B_Rep_2_FDR20.parquet`
- `experiment_XYZ_run_003_FDR50.parquet`  
- `20240315_sample_ABC_method1_FDR1.parquet`
- `QC_Standard_Run_042_Method_XYZ-789_FDR50.parquet`
- `HPLC90_Protocol3_Bio1_FDR20.parquet`

The tool automatically extracts meaningful method names from ANY filename!

### 3. Launch and Use
```bash
./start_peptidia.sh
```

Your dataset will appear automatically in the interface! ✨

## 📋 File Requirements

- **Format**: `.parquet` files from DIA-NN analysis
- **FDR Levels**: Include FDR_1, FDR_20, and/or FDR_50 as needed
- **Structure**: Must follow the folder structure above
- **Naming**: Any naming convention works - the tool is completely universal

## 🎨 Optional Customization

Add `data/YourDatasetName/dataset_info.json` for custom display and ground truth matching:

### Basic Configuration (Display Only):
```json
{
  "icon": "🔬",
  "instrument": "Your mass spectrometer name",
  "description": "Brief description of your dataset"
}
```

### Advanced Configuration (With Ground Truth Matching):

**Option 1: Use All Ground Truth Files (Simplest)**
```json
{
  "icon": "🔬",
  "instrument": "Your mass spectrometer name", 
  "description": "Brief description of your dataset",
  "ground_truth_mapping": {
    "_strategy": "use_all_ground_truth",
    "_note": "Any method uses all ground truth files from this dataset"
  }
}
```

**Option 2: Specific Pattern Matching (Advanced)**
```json
{
  "icon": "🔬",
  "instrument": "Your mass spectrometer name",
  "description": "Brief description of your dataset", 
  "ground_truth_mapping": {
    "_strategy": "pattern_matching",
    "_pattern_rules": {
      "001": "GroundTruth_A",
      "002": "GroundTruth_B",
      "003": "GroundTruth_C"
    },
    "_note": "Method_001 uses GroundTruth_A, Method_002 uses GroundTruth_B, etc."
  }
}
```

### Real Examples:

**ASTRAL (Specific Mapping):**
```json
{
  "ground_truth_mapping": {
    "_pattern_rules": {
      "001": "RR-073", "002": "RR-074", "003": "RR-075",
      "004": "RR-076", "005": "RR-077", "006": "RR-078", 
      "007": "RR-080", "008": "RR-081", "009": "RR-082", "010": "RR-082"
    }
  }
}
```

**HEK (Use All Files):**
```json
{
  "ground_truth_mapping": {
    "_strategy": "use_all_ground_truth"
  }
}
```

## 🧪 Examples

### Example 1: Simple Lab Setup
```
data/SmithLab/
├── short_gradient/FDR_1/experiment_001_FDR1.parquet
├── short_gradient/FDR_20/experiment_001_FDR20.parquet  
└── long_gradient/FDR_1/ground_truth_001_FDR1.parquet
```

### Example 2: Complex Research Project
```
data/ProteomicsCore/
├── short_gradient/
│   ├── FDR_1/LC120_MethodA_Bio1_FDR1.parquet
│   ├── FDR_20/LC120_MethodA_Bio1_FDR20.parquet
│   └── FDR_50/LC120_MethodA_Bio1_FDR50.parquet
└── long_gradient/FDR_1/LC480_LongGrad_Bio1_FDR1.parquet
```

## 🔄 Migration from Old Structure

If you have data in different folder structures, just reorganize into the standard layout above. The tool automatically handles the rest!

## ❓ Troubleshooting

- **No files detected**: Check folder structure matches exactly
- **Method names look strange**: This is normal - any unique identifier works
- **Missing baseline**: Make sure you have `short_gradient/FDR_1/` files
- **No ground truth**: Ensure `long_gradient/FDR_1/` files exist

## 🎯 What Makes This Universal?

The Enhanced Peptide Validator now uses **intelligent pattern recognition** and **configuration-based ground truth matching**:

### Data Discovery:
1. **Scientific Method Patterns**: Detects MS30-DIA7-5, LC120-Protocol1, etc.
2. **Replicate Series**: Handles sample_001, experiment_A_rep_2, etc.  
3. **Unique Identifiers**: Extracts codes like XYZ-789, ABC123, etc.
4. **Smart Segmentation**: Analyzes filename structure intelligently
5. **Fallback Logic**: Always generates a usable method name

### Ground Truth Matching:
1. **Configuration-Based**: Each dataset can specify how methods match to ground truth
2. **Flexible Strategies**: Use all files or specific pattern matching
3. **No Code Changes**: All configuration via `dataset_info.json` files
4. **Smart Defaults**: Works automatically even without configuration

## 📞 Support

If you encounter issues, the tool provides helpful error messages. The goal is to make this work with **any research group's naming conventions** without code modifications!

---

**🎉 Ready to analyze your data? Follow the 3 steps above and start discovering additional peptides in your DIA experiments!**