<h1 align="center">
  HIAE
</h1>
<h4 align="center">HIAE: Hyper-Relational Interaction Aware Embedding for Link Prediction</h4>
<h2 align="center">
  Overview of HIAE
  <img align="center"  src="./overview.png" alt="...">
</h2>

### Dependencies

- PyTorch 1.x and Python 3.x.

### Dataset:

- We use JF17K,  WikiPeople, JF17K_clean, WikiPeople_clean and WD50K datasets for evaluation. 

### Training model from scratch:

- To start training **InteractE** run:

  ```shell
  # JF17K
  python run.py --data JF17K --gpu 0 --name fb15k_237_run
  
  # WN18RR
  python interacte.py --data WN18RR --batch 256 --train_strategy one_to_n --feat_drop 0.2 --hid_drop 0.3 --perm 4 --ker_sz 11 --lr 0.001
