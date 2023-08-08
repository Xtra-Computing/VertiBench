# Satellite Dataset

## 1. Prerequisite

1. **Download** the `satellite.zip` file from [here](https://drive.google.com/drive/folders/1Ti73Doy7xW0BRv2D8FHZFqSlzZWfd2gj)
2. **Verify** the file integrity using the following command:
```
md5sum satellite.zip
```
Make sure that the result matches the value provided in `satellite.zip.md5sum`.

3. **Unzip** the `satellite.zip`, If the file integrity check is successful, the zip file can be extracted using any standard unzipping tool. After extraction, the dataset's file structure will be as follows:

```bash
   .
├── satellite_party0_test.csv
├── satellite_party0_test.pkl
├── satellite_party0_train.csv
├── satellite_party0_train.pkl
├── satellite_party1_test.csv
├── ...
└── satellite_party15_train.pkl
```

## 2. File Structure

The dataset is organized into `16` parties, each containing test and training files in CSV formats. 

The naming convention follows the pattern `satellite_partyX_test.csv` and `satellite_partyX_train.csv`, where `X` ranges from `0` to `15`.

### 2.1 CSV File Format
Each CSV file consists of a header and the following columns (total 324,534 columns):

`id`: ID

`y`: Label

`x0`: Feature 0

`x1`: Feature 1

...

`x324531`: Feature 324,531

