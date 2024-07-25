# Satellite Dataset

The Satellite dataset encompasses Point of Interest (POI) data, each associated with one or more Areas of Interest (AOI).

Every AOI incorporates a unique location identifier, a land type, and **16 low-resolution**, **13-channel** images (13 spectral bands, up to 10 m/pixel), each taken during a satellite visit to the location. 

The Satellite dataset encompasses **`four land types`** as labels, namely `Amnesty POI` (4.8%), `ASMSpotter` (8.9%), `Landcover` (61.3%), and `UNHCR` (25.0%), making the task a **`4-class classification problem`** of **`3927 locations`**.



<img width="1222" alt="all bands" src="https://github.com/JerryLife/VertiBench/assets/14367694/4386ed55-76b5-4282-a374-8a03c9da509c">

See `Satellite.ipynb` for more details.



## 1. Prerequisite

1. **Download** the `satellite.zip` file from [here](https://drive.google.com/drive/folders/1Ti73Doy7xW0BRv2D8FHZFqSlzZWfd2gj)
2. **Verify** the file integrity using the following command:

   (Make sure that the result matches the value provided in `satellite.zip.md5sum`.)
```
md5sum satellite.zip # 
```


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
<img width="1126" alt="file" src="https://github.com/JerryLife/VertiBench/assets/14367694/f2604d16-b3a5-49b6-a919-494bf42cdad5">



### 2.1 CSV File Format
Each CSV file consists of a header and the following columns (total 324,534 columns):

```
id, y, x0, x1, ..., x324531
```

For example the `satellite_party0_test.csv`
<img width="1129" alt="csv" src="https://github.com/JerryLife/VertiBench/assets/14367694/2920bed6-ba7c-4f39-88bc-d8746d72718b">

## 3. Convert CSV to Image

You can use the CSV data directly, or convert it to an image, column `x0` to `x324531` represents the `13-channel` `158x158 pixel` image.
the pixel values of each image are scaled to integer values within the range of [0, 255]
`324532 = 13 x 158 x 158`

See `Satellite.ipynb` for more details.

## 4. Classification Task Details

- **algorithm:** `SplitResNet` (from torchvision.models import resnet18)
- **#classes:** `4` (four land types)
- **#parties:** `16`
- **#epochs:** `50`
- **metric:** `accuracy`
- **learning_rate:** `1e-5`
- **batch_size:** `32`
- **loss_function:** `Cross-Entropy`
- **early_stopping:** `None`
- **channel:** `13`
- **kernel_size :** `9`
- **out_activation:** `None`
- **agg_hidden:** `[1000, out_dim]`
- **out_dim:** `4` (same as #classes)
- **optimizer:** `Adam`
- **weight_decay:** `1e-5`
- **lr_scheduler**: `StepLR(step_size=10, gamma=0.5)`

```py
class SplitResNet(nn.Module):
    def __init__(self, n_parties, channels, kernel_size=9, agg_hidden=None, out_activation=None):
        super().__init__()
        self.n_parties = n_parties
        self.out_activation = out_activation
        self.local_resnet_list = nn.ModuleList()
        local_output_dims = []
        for i in range(self.n_parties):
            resnet = resnet18(weights=None)
            local_output_dims.append(resnet.fc.in_features)
            resnet.fc = nn.Identity()
            resnet.conv1 = nn.Conv2d(channels, 64, kernel_size, stride=2, padding=3, bias=False)
            self.local_resnet_list.append(resnet)
            print("local output dims", local_output_dims)
        self.cut_dim = sum(local_output_dims)
        if agg_hidden is None:
            self.agg_hidden = [100, 1]
        else:
            self.agg_hidden = agg_hidden
        self.agg_mlp = MLP(self.cut_dim, self.agg_hidden)
        if out_activation is None:
            self.out_activation = nn.Identity()
        else:
            self.out_activation = out_activation

    def forward(self, Xs):
        # print("Shape of Xs: ", [Xi.shape for Xi in Xs])
        local_outputs = [resnet(Xi) for resnet, Xi in zip(self.local_resnet_list, Xs)]
        agg_input = torch.cat(local_outputs, dim=1)
        agg_output = self.agg_mlp(agg_input)
        return self.out_activation(agg_output)
```

## License

The VertiBench Satellite Dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material for any purpose, even commercially, as long as you provide appropriate credit, link to the license, and indicate if changes were made.

Please refer to the full license text for more detailed information.
