import argparse

def parser_train():
    parser = argparse.ArgumentParser(description='Args for Training.')
    
    # basic
    parser.add_argument('--desc', type=str, default='none', help='Description of experiment. It will be used to name directories.')
    parser.add_argument('--save-dir', type=str, default='./save')
    parser.add_argument('--seed', type=int, default=2333)
    
    # whole training
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=128, help='Batch size for testing.') 
    parser.add_argument('--ensemble-iter-eval',  type=int, default=10, help='Number of ensemble while evaluating, helping choose best epoch')
     
    # data 
    parser.add_argument('--data-dir', type=str, default='./datasets')
    parser.add_argument('--data', type=str, default='cifar10', choices=['mnist','cifar10','cifar100'], help='Data to use.')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--npc-train', default='all', help='Number of training samples per class, int or all.') 
    
    # augmentation
    parser.add_argument('--aug-train-inplace',  type=str, default="augmix", help='Data augmentation for training, replacing clean') # choices=["gaublur-1-3", "elastic", "augmix", "augmix-6-4","randaug", "none"], 
    parser.add_argument('--aug-train',  type=str, default="augmix", help='Data augmentation for training') # choices=["gaublur-1-3", "elastic", "augmix", "augmix-6-4","randaug", "none"], 
    parser.add_argument('--severity-eval',  type=int, choices=[1,2,3,4,5], default=5, help='Data augmentation severity for evaluating')

    # attack
    parser.add_argument('--atk-train', type=str, choices=['fgsm', 'linf-pgd', 'fgm', 'l2-pgd', 'linf-df', 'l2-df', 'linf-apgd', 'l2-apgd', 'none'], 
                        default='linf-pgd', help='Type of attack for training.')
    parser.add_argument('--atk-eval', type=str, choices=['fgsm', 'linf-pgd', 'fgm', 'l2-pgd', 'linf-df', 'l2-df', 'linf-apgd', 'l2-apgd', 'none'], 
                        default='linf-pgd', help='Type of attack for evaluating.')
    parser.add_argument('--attack-eps', type=float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')

    # model
    parser.add_argument('--backbone', type=str, choices=['resnet-18', 'wideresnet-16-4', 'preresnet-18'], default="resnet-18")
    parser.add_argument('--protocol', type=str,  default="ladiff-augdiff") # choices=['standard', 'fixdiff', 'ladiff-augdiff', 'ladiff-oridiff']
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')
    parser.add_argument('--resume-file', type=str, default=None, help='Resumed file name.')
    parser.add_argument('--save-freq', type=int, default=50, help='Save per epochs.') 
    
    # C optimizer
    parser.add_argument('--optimizerC', type=str, default='sgd', help='Choice for optimizerC.')
    parser.add_argument('--lrC', type=float, default=0.01, help='Learning rate for optimizer.')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=['cosanlr', 'mslr', 'piecewise', "none"], default='cosanlr', help='Type of scheduler.')
    parser.add_argument('--warm', type=int, default=0)

    # Diff optimizer
    parser.add_argument('--optimizerDiff', type=str, default='adam', help='Choice for optimizerD.')
    parser.add_argument('--lrDiff', type=float, default=0.1, help='Learning rate for optimizer.')

    args = parser.parse_args()
    return args


def parser_test():
    parser = argparse.ArgumentParser(description='Args for Testing.')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--main_task', type=str, choices=["ood","adv"])
    parser.add_argument('--severity',  type=int, choices=[0,1,2,3,4,5], help='Data augmentation severity for testing')
    parser.add_argument('--type',  type=str, choices=['c15','c19'], help='Data augmentation type for testing')
    parser.add_argument('--threat',  type=str, choices=['linf','l2'])
    parser.add_argument('--load_ckpt', type=str)
    args = parser.parse_args()
    return args