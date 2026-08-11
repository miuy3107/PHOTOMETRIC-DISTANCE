# Star Distance Determination

This project explores stellar distance estimation with photometry and machine learning. It uses the HYG star catalog to prepare a clean dataset, verify the distance-modulus formula, and train Random Forest models that predict a star's absolute magnitude. That prediction can then be combined with apparent magnitude to estimate distance.

## Project workflow

```text
hyg_v44.csv.gz
     |
     v
clean_data.py  -->  hygdata_clean.csv
     |                    |
     |                    +--> photometric_distance.py --> hygdata_with_predictions.csv
     |                    |
     |                    +--> ml.py  --> ml_test_results.csv
     |                    |
     |                    +--> ml2.py --> ml_test_results_v2.csv
```

## Scripts

| File | Purpose |
| --- | --- |
| `clean_data.py` | Selects useful HYG catalog columns, removes incomplete or invalid records, and saves `hygdata_clean.csv`. |
| `photometric_distance.py` | Calculates distance from apparent and absolute magnitude and compares it with the catalog distance. |
| `ml.py` | Trains a baseline Random Forest model using only the color index (`ci`) to predict absolute magnitude (`absmag`). |
| `ml2.py` | Trains an improved Random Forest model using color index plus spectral luminosity class, after removing variable stars. |

## Data

Place `hyg_v44.csv.gz` in the project root before running the cleaning script. The cleaned dataset includes these relevant columns:

| Column | Meaning |
| --- | --- |
| `proper` | Common or proper star name, when available. |
| `mag` | Apparent magnitude. |
| `absmag` | Absolute magnitude. |
| `ci` | Color index (B-V). |
| `spect` | Spectral type. |
| `dist` | Distance from parallax, in parsecs. |
| `var`, `var_min`, `var_max` | Variable-star identifier and brightness range fields. |

## Installation

Use Python 3. Install the required packages:

```bash
pip install pandas numpy scikit-learn
```

## Run the project

Run the scripts from the project directory, in this order:

```bash
python clean_data.py
python photometric_distance.py
python ml.py
python ml2.py
```

The later scripts require `hygdata_clean.csv`, which is produced by `clean_data.py`.

## Distance formula

The project uses the distance modulus:

```text
distance (pc) = 10 ^ ((apparent magnitude - absolute magnitude + 5) / 5)
```

`photometric_distance.py` uses the catalog's `absmag`, while the machine-learning scripts first estimate `absmag` and then apply this formula.

## Models

The baseline model in `ml.py` predicts absolute magnitude from the B-V color index (`ci`). The improved model in `ml2.py` also extracts luminosity class from the spectral type and one-hot encodes it.

For example:

- `G2V` is a main-sequence star.
- `K1III` is a giant.
- `DA2` is a white dwarf.

Stars with a confirmed variable-star label or a brightness range greater than 0.1 magnitudes are excluded by `ml2.py`, because their brightness is not stable enough for a single photometric measurement to be representative.

## Output files

| File | Contents |
| --- | --- |
| `hygdata_clean.csv` | Cleaned input data for analysis and modeling. |
| `hygdata_with_predictions.csv` | Formula-based distance predictions and error measurements. |
| `ml_test_results.csv` | Test-set predictions from the baseline model. |
| `ml_test_results_v2.csv` | Test-set predictions from the improved model. |

## Notes

- The catalog's absolute magnitude is derived from parallax distance, so the direct formula check is expected to be very accurate.
- The model evaluation uses an 80/20 train-test split with `random_state=42` for reproducible results.
- This is an educational experiment; photometric distance estimates are affected by factors such as extinction, stellar classification uncertainty, and measurement error.
