from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_scheduler(args, opt):
    if args.scheduler =="cosanlr":
        return CosineAnnealingLR(opt, T_max=args.epoch)
    elif args.scheduler =="mslr":
        return MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2) #60, 120, 160 #200, 250, 300, 350
    elif args.scheduler =="none":
        return None
    else:
        raise
