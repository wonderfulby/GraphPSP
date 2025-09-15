import argparse

def arg_parse():
        #-定义一个函数 `arg_parse`，用于解析命令行参数
        parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
        parser.add_argument('--DS', dest='DS',default='COX2' ,help='Dataset')
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                help='Number of graph convolution layers before each pooling')
        parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
                help='')
        parser.add_argument('--preprocess', dest='preprocess', default=False)
        parser.add_argument('--fold_idx', type=int, default=8,
                            help='the index of fold in 10-fold validation. Should be less then 10.')
        parser.add_argument('--ptb', type=float, default=0.3,
                            help="noise ptb_rate")
        parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                            help='type of noises')
        parser.add_argument('--seed', type=int, default=0,
                            help='random seed for splitting the dataset into 10 (default: 0)')
        parser.add_argument('--num_filtrations', type=int, default=8,
                            help='过滤函数的数量 (默认值: 8)')
        parser.add_argument('--filtration_hidden', type=int, default=32,
                            help='过滤隐藏层的维度 (默认值: 16)')
        parser.add_argument('--out_ph_dim', type=int, default=16,
                            help='拓扑特征输出维度 (默认值: 32)')
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--num_node_features", type=int, default=16)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument('--final_dropout', type=float, default=0.5,
                            help='final layer dropout (default: 0.5)')
        parser.add_argument('--aug1', type=str, default='identity')
        parser.add_argument('--aug2', type=str, default='feature_drop')
        parser.add_argument('--patience', type=int, default=5)
        parser.add_argument('--use_select_sim', type=bool, default=False)
        parser.add_argument('--t', type=float, default=0.3)
        parser.add_argument('--degree_as_tag', action="store_true",
                            help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')

        return parser.parse_args()

