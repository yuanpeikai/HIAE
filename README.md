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

- To start training **InteractE** run:

  ```shell
  # FB15k-237
  python interacte.py --data FB15k-237 --gpu 0 --name fb15k_237_run
  
  # WN18RR
  python interacte.py --data WN18RR --batch 256 --train_strategy one_to_n --feat_drop 0.2 --hid_drop 0.3 --perm 4 --ker_sz 11 --lr 0.001
  
  # YAGO03-10
  python interacte.py --data YAGO3-10 --train_strategy one_to_n  --feat_drop 0.2 --hid_drop 0.3 --ker_sz 7 --num_filt 64 --perm 2
  ```
  - `data` indicates the dataset used for training the model. Other options are `WN18RR` and `YAGO3-10`.
  - `gpu` is the GPU used for training the model.
  - `name` is the provided name of the run which can be later used for restoring the model.
  - Execute `python interacte.py --help` for listing all the available options.
