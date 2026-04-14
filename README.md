# Train and Validate ML Battery Aging Models

## Publication
If you find this work useful please give credits to the authors by citing:
* Shahsavari S, Immonen E, Haghbayan H, Plosila J. **A Novel Approach for Battery State-of-Health Estimation Using Convolutional Auto-Encoders**. In2025 European Control Conference (ECC) 2025 Jun 24 (pp. 2433-2439) ([PDF](https://ieeexplore.ieee.org/abstract/document/11186898))

```
@inproceedings{shahsavari2025novel,
  title={A Novel Approach for Battery State-of-Health Estimation Using Convolutional Auto-Encoders},
  author={Shahsavari, Sajad and Immonen, Eero and Haghbayan, Hashem and Plosila, Juha},
  booktitle={2025 European Control Conference (ECC)},
  pages={2433--2439},
  year={2025},
  organization={IEEE}
}
```

## Install
Create and enable a virtual environment, then install the requirement libraries:
```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Run training and validation scripts
For each dataset, you can find the corresponding scripts to train AutoEncoder and SoH prediction models in `scripts` direcotry. For the NASA dataset, for example, you can run the following:
```bash
  cd scripts/nasa
  ./run.sh
  ```

There are various options for the `run.sh` script:
```bash
  ./run.sh --exp-name "test" train-cells "RW01" --valid-cells "RW09" \
            --window-length 32768 --window-step 32768 --latent-size 256 \
            --nj 8 -- data-dir "DATA_DIR" --epochs 100 --lr 0.001
  ```


