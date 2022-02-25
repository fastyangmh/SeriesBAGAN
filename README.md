# SeriesBAGAN

## install
```bash
pip install -r requirements.txt
```

## dataset
```
data
└── FCS_data
    ├── EU_label.xlsx
    ├── EU_marker_channel_mapping.xlsx
    ├── fcs_3
    │   ├── flowrepo_covid_EU_007_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_ICU_002_A ST3 230420_052_Live_cells.fcs
    │   ├── flowrepo_covid_EU_009_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_005_A ST3 210420_080_Live_cells.fcs
    │   └── flowrepo_covid_EU_041_flow_001
    │       └── export_COVID19 samples 23_04_20_ST3_COVID19_W_024_O ST3 230420_005_Live_cells.fcs
    ├── raw_fcs
    │   ├── flowrepo_covid_EU_002_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_HC_005 ST3 230420_016_Live_cells.fcs
    │   ├── flowrepo_covid_EU_003_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_HC_006 ST3 230420_015_Live_cells.fcs
    │   ├── flowrepo_covid_EU_004_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_HC_007 ST3 230420_014_Live_cells.fcs
    │   ├── flowrepo_covid_EU_005_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_HC_008 ST3 230420_013_Live_cells.fcs
    │   ├── flowrepo_covid_EU_006_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_HC_009 ST3 230420_012_Live_cells.fcs
    │   ├── flowrepo_covid_EU_007_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_ICU_002_A ST3 230420_052_Live_cells.fcs
    │   ├── flowrepo_covid_EU_008_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_ICU_003_A ST3 230420_049_Live_cells.fcs
    │   ├── flowrepo_covid_EU_009_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_005_A ST3 210420_080_Live_cells.fcs
    │   ├── flowrepo_covid_EU_010_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_ICU_006_A ST3 230420_046_Live_cells.fcs
    │   ├── flowrepo_covid_EU_011_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_ICU_012_A ST3 230420_043_Live_cells.fcs
    │   ├── flowrepo_covid_EU_012_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_013_A ST3 210420_078_Live_cells.fcs
    │   ├── flowrepo_covid_EU_013_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_014_A ST3 210420_073_Live_cells.fcs
    │   ├── flowrepo_covid_EU_014_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_015_A ST3 210420_069_Live_cells.fcs
    │   ├── flowrepo_covid_EU_015_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_023_A ST3 210420_064_Live_cells.fcs
    │   ├── flowrepo_covid_EU_016_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_ICU_025_A ST3 230420_039_Live_cells.fcs
    │   ├── flowrepo_covid_EU_017_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_027_A ST3 210420_061_Live_cells.fcs
    │   ├── flowrepo_covid_EU_018_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_029_A ST3 210420_059_Live_cells.fcs
    │   ├── flowrepo_covid_EU_019_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_031_A ST3 210420_055_Live_cells.fcs
    │   ├── flowrepo_covid_EU_020_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_036_A ST3 210420_053_Live_cells.fcs
    │   ├── flowrepo_covid_EU_021_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_039_A ST3 210420_051_Live_cells.fcs
    │   ├── flowrepo_covid_EU_022_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_043_A ST3 210420_049_Live_cells.fcs
    │   ├── flowrepo_covid_EU_023_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_ICU_045_A ST3 210420_045_Live_cells.fcs
    │   ├── flowrepo_covid_EU_030_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_001_O ST3 230420_037_Live_cells.fcs
    │   ├── flowrepo_covid_EU_031_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_010_O ST3 230420_033_Live_cells.fcs
    │   ├── flowrepo_covid_EU_032_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_011_O ST3 230420_032_Live_cells.fcs
    │   ├── flowrepo_covid_EU_033_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_012_O ST3 230420_031_Live_cells.fcs
    │   ├── flowrepo_covid_EU_034_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_014_O ST3 230420_029_Live_cells.fcs
    │   ├── flowrepo_covid_EU_035_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_015_O ST3 230420_028_Live_cells.fcs
    │   ├── flowrepo_covid_EU_036_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_016_O ST3 230420_027_Live_cells.fcs
    │   ├── flowrepo_covid_EU_037_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_017_O ST3 230420_026_Live_cells.fcs
    │   ├── flowrepo_covid_EU_038_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_018_O ST3 230420_025_Live_cells.fcs
    │   ├── flowrepo_covid_EU_039_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_020_O ST3 230420_007_Live_cells.fcs
    │   ├── flowrepo_covid_EU_040_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_022_O ST3 230420_006_Live_cells.fcs
    │   ├── flowrepo_covid_EU_041_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_024_O ST3 230420_005_Live_cells.fcs
    │   ├── flowrepo_covid_EU_042_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_025_O ST3 230420_004_Live_cells.fcs
    │   ├── flowrepo_covid_EU_043_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_026_O ST3 230420_003_Live_cells.fcs
    │   ├── flowrepo_covid_EU_044_flow_001
    │   │   └── export_COVID19 samples 23_04_20_ST3_COVID19_W_029_O ST3 230420_002_Live_cells.fcs
    │   ├── flowrepo_covid_EU_046_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_W_042_O ST3 210420_032_Live_cells.fcs
    │   ├── flowrepo_covid_EU_047_flow_001
    │   │   └── export_COVID19 samples 21_04_20_ST3_COVID19_W_043_O ST3 210420_031_Live_cells.fcs
    │   └── flowrepo_covid_EU_048_flow_001
    │       └── export_COVID19 samples 23_04_20_ST3_COVID19_W_046_O ST3 230420_011_Live_cells.fcs
    ├── raw_fcs.zip
    └── test_description.pdf
```

## feature selection

Select features according to column name "use" in file EU_marker_channel_mapping.xlsx

## run
```bash
python VAE.py
python GAN.py
```