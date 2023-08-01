# Stop-loss adjusted labels for machine learning-based trading of risky assets

This is the origin Pytorch implementation of Informer in the following paper: Stop-loss adjusted labels for machine learning-based trading of risky assets


**News**(July 31, 202023):  Accepted to [Finance Research Letters](https://www.sciencedirect.com/journal/finance-research-letters) 2023.
 

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data.
3. Run Stop-loss adjusted labels.

### Reproduce with Docker

To easily reproduce the results using Docker, conda and Make,  you can follow the next steps:
1. Initialize the docker image using: `make init`. 
2. Download the datasets using: `make data`.
3. Download the datasets using: `make coin data`.
4. Run each script in `runfile/` using `make run_module module="bash runfile/btc_runfile.sh"` for each script.
5. Alternatively, run all the scripts at once:
```
for file in `ls scripts`; do make run_module module="bash runfile/runfile"; done
```

## Baselines

We will keep adding Predicting movements of asset prices models to expand this repo:

- [x] SVM
- [x] KNN
- [x] MLP
- [x] Catboost
- [x] Random Forest
- [x] Extra tree

## Risk measrue

- [x] MDD
- [x] VAR, CVAR
- [x] Adjusted Sharpe Ratio
- [x] Sortino Ratio


## Citation

If you find this repo useful, please cite our paper. 

```
Will be update
```

## Contact

If you have any questions or want to use the code, please contact `yoontae@unist.ac.kr`

