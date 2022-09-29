from helper import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from data_loader import *
from model import *


class Main(object):

    def __init__(self, params):

        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')

            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        self.p.device = self.device
        self.load_data()
        self.model = self.add_model()
        self.optimizer = self.add_optimizer(self.model.parameters())

    def load_data(self):

        ent_set, rel_set = OrderedSet(), OrderedSet()
        ent_set.add('no_ent')
        rel_set.add('no_rel')
        if self.p.dataset == 'jf17k' or self.p.dataset == 'jf17k_clean':
            splits = ['train', 'test']
        else:
            splits = ['train', 'test', 'valid']

        for split in splits:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                line = line.rstrip("\n")
                text = line.strip().split(',')
                ent = text[::2]
                rel = text[1::2]
                for t in ent:
                    ent_set.add(t)
                for t in rel:
                    rel_set.add(t)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        # self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for rel, idx in self.rel2id.items()})
        rel2id_tail = {}
        for rel, idx in self.rel2id.items():
            if idx != 0:
                rel2id_tail[rel + '_reverse'] = len(self.rel2id) + idx - 1
        self.rel2id.update(rel2id_tail)
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in splits:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                line = line.rstrip("\n")
                text = line.strip().split(',')
                sub, rel, obj = self.ent2id[text[0]], self.rel2id[text[1]], self.ent2id[text[2]]
                N = (len(text) - 3) // 2

                quals = []
                for i, v in enumerate(text[3::]):
                    if (i % 2 == 0):
                        quals.append(self.rel2id[v])
                    else:
                        quals.append(self.ent2id[v])
                quals = quals + [0] * (self.p.max_quals - N * 2)

                self.data[split].append((sub, rel, obj, N, quals))

                if split == 'train' or split == 'valid':
                    sr2o[(sub, rel, N, *quals)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel, N, *quals)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test']:
            for sub, rel, obj, N, quals in self.data[split]:
                sr2o[(sub, rel, N, *quals)].add(obj)
                sr2o[(obj, rel + self.p.num_rel, N, *quals)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        self.triples = ddict(list)

        for (sub, rel, N, *quals), obj in self.sr2o.items():
            arr_N = np.zeros(self.p.max_quals // 2)
            i = N
            while (i != 0):
                arr_N[i - 1] = 1
                i = i - 1
            self.triples['train'].append(
                {'triple': (sub, rel, -1), 'quals': quals, 'label': self.sr2o[(sub, rel, N, *quals)], 'N': arr_N})

        for split in ['test']:
            for sub, rel, obj, N, quals in self.data[split]:
                rel_inv = rel + self.p.num_rel
                arr_N = np.zeros(self.p.max_quals // 2)
                i = N
                while (i != 0):
                    arr_N[i - 1] = 1
                    i = i - 1
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'quals': quals, 'label': self.sr2o_all[(sub, rel, N, *quals)],
                     'N': arr_N})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'quals': quals, 'label': self.sr2o_all[(obj, rel_inv, N, *quals)],
                     'N': arr_N})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {}

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

    def add_model(self):
        """
        Creates the computational graph

        Parameters
        ----------

        Returns
        -------
        Creates the computational graph for model and initializes it

        """

        if self.p.model_name == 'InteractE':
            model = InteractE(self.p)
        elif self.p.model_name == 'Transformer_Triple':
            model = Transformer_Triple(self.p)
        elif self.p.model_name == 'Transformer_Query':
            model = Transformer_Query(self.p)
        model.to(self.device)

        params = [value.numel() for value in model.parameters()]
        print(params)
        print(f'model_para : {np.sum(params)}')

        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        if self.p.opt == 'adam':
            return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
        else:
            return torch.optim.SGD(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        triples:	The triples used for this split
        labels:		The label for each triple
        """
        triple, label, quals, N = [_.to(self.device) for _ in batch]
        return triple[:, 0], triple[:, 1], triple[:, 2], label, quals, N

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val_mrr = state['best_val']['mrr']
        self.best_val = state['best_val']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch=0):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)
        self.logger.info(
            '[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'],
                                                                                 results['right_mrr'], results['mrr']))
        self.logger.info(
            '[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mr'],
                                                                                results['right_mr'], results['mr']))
        self.logger.info(
            '[Epoch {} {}]: hits@10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split,
                                                                                     results['left_hits@10'],
                                                                                     results['right_hits@10'],
                                                                                     results['hits@10']))
        self.logger.info(
            '[Epoch {} {}]: hits@5: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split,
                                                                                    results['left_hits@5'],
                                                                                    results['right_hits@5'],
                                                                                    results['hits@5']))

        self.logger.info(
            '[Epoch {} {}]: hits@3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split,
                                                                                    results['left_hits@3'],
                                                                                    results['right_hits@3'],
                                                                                    results['hits@3']))
        self.logger.info(
            '[Epoch {} {}]: hits@1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split,
                                                                                    results['left_hits@1'],
                                                                                    results['right_hits@1'],
                                                                                    results['hits@1']))
        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label, quals, N = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel, quals, N)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                label[:, 0] = 1
                pred = torch.where(label.byte(), torch.zeros_like(pred), pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                # if step % 100 == 0:
                #     self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results

    def run_epoch(self, epoch):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        start_time = time.time()

        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            # sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')
            sub, rel, obj, label, quals, N = self.read_batch(batch, 'train')

            pred = self.model.forward(sub, rel, quals, N)
            loss = self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:	time:{}  Training Loss:{:.4}\n'.format(epoch, time.time() - start_time, loss))
        print('当前学习率：{}'.format(self.optimizer.param_groups[0]["lr"]))
        return loss

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0.
        val_mrr = 0
        save_path = os.path.join('./torch_saved', self.p.name)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', verbose=1, factor=0.1,
                                                               patience=30, min_lr=0.0001)
        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch(epoch)
            val_results = self.evaluate('test', epoch)
            scheduler.step(val_results['mrr'])
            # val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
            self.logger.info(
                '[best Epoch {}]:  Training Loss: {:.5},  best MRR: {:.5}, \n\n\n'.format(self.best_epoch, train_loss,
                                                                                          self.best_val_mrr))

        # Restoring model corresponding to the best validation performance and evaluation on test data
        self.logger.info('Loading best model, evaluating on test data')
        self.load_model(save_path)
        self.evaluate('test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser For Arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset and Experiment name
    parser.add_argument('--data', dest="dataset", default='jf17k', help='Dataset to use for the experiment')
    parser.add_argument("--name", default='testrun_' + str(uuid.uuid4())[:8], help='Name of the experiment')
    # parser.add_argument("--name", default='testrun_a226c109', help='Name of the experiment')
    parser.add_argument('--restore', dest="restore", action='store_true',
                        help='Restore from the previously saved model')

    # Training parameters
    parser.add_argument("--gpu", type=str, default='0', help='GPU to use, set -1 for CPU')

    parser.add_argument("--opt", type=str, default='adam', help='Optimizer to use for training')

    parser.add_argument('--batch', dest="batch_size", default=128, type=int, help='Batch size')
    parser.add_argument("--l2", type=float, default=0.0, help='L2 regularization')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning Rate')
    parser.add_argument("--epoch", dest='max_epochs', default=2000, type=int, help='Maximum number of epochs')
    parser.add_argument("--num_workers", type=int, default=0, help='Maximum number of workers used in DataLoader')
    parser.add_argument('--seed', dest="seed", default=40, type=int, help='Seed to reproduce results')
    parser.add_argument('-max_quals', dest="max_quals", default=8, type=int, help='')
    # Model parameters
    parser.add_argument("--lbl_smooth", dest='lbl_smooth', default=0.1, type=float,
                        help='Label smoothing for true labels')
    parser.add_argument("--embed_dim", type=int, default=200,
                        help='Embedding dimension for entity and relation, ignored if k_h and k_w are set')
    parser.add_argument('--bias', dest="bias", action='store_true', help='Whether to use bias in the model')
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--agg_func", type=str, default='HyperCrossE', help='')
    parser.add_argument("--method", type=str, default='co-aware', help='')
    parser.add_argument("--model_name", type=str, default='InteractE', help='')

    # Logging parameters
    parser.add_argument('--logdir', dest="log_dir", default='./log/', help='Log directory')
    parser.add_argument('--config', dest="config_dir", default='./config/', help='Config directory')

    # interacte
    parser.add_argument('--hid_drop', dest="hid_drop", default=0.5, type=float, help='Dropout for Hidden layer')
    parser.add_argument('--feat_drop', dest="feat_drop", default=0.5, type=float, help='Dropout for Feature')
    parser.add_argument('--inp_drop', dest="inp_drop", default=0.2, type=float, help='Dropout for Input layer')
    parser.add_argument('--channel', dest="channel", default=96, type=int, help='Number of out channel')
    parser.add_argument("--filter_size", type=int, default=9)
    parser.add_argument('--iperm', dest="iperm", default=1, type=int, help='')
    parser.add_argument('-ik_w', dest="ik_w", default=10, type=int, help='Width of the reshaped matrix')
    parser.add_argument('-ik_h', dest="ik_h", default=20, type=int, help='Height of the reshaped matrix')

    # # transformer
    parser.add_argument('-heads', dest="heads", default=4, type=int, help='')
    parser.add_argument('-hidden_dim', dest="hidden_dim", default=512, type=int, help='')
    parser.add_argument('--hidden_drop', dest="hidden_drop", default=0.1, type=float, help='')
    parser.add_argument('-layers', dest="layers", default=2, type=int, help='')

    torch.set_num_threads(2)

    args = parser.parse_args()

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Main(args)
    model.fit()

