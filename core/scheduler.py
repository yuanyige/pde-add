from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR
import numpy as np
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
        return ("cosanlr", CosineAnnealingLR(opt, T_max=args.epoch))
    elif args.scheduler =="mslr":
        return ("mslr", MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2))
    elif args.scheduler == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, args.epoch * 2 // 5, args.epoch], [0, args.lrC, 0])[0]
        return ("cyclic", lr_schedule)
    elif args.scheduler == 'piecewise':
        def lr_schedule(t):
            if args.epoch == 0:
                return args.lrC
            if t / args.epoch < 0.34:#0.6:
                return args.lrC
            elif t / args.epoch < 0.67:#< 0.9:
                return args.lrC / 10.
            else:
                return args.lrC / 100.
        return ("piecewise", lr_schedule)
    elif args.scheduler =="none":
        return None
    else:
        raise

# def get_lr_schedule(lr_schedule_type, n_epochs, lr_max):
#     if lr_schedule_type == 'cyclic':
#         lr_schedule = lambda t: np.interp([t], [0, n_epochs * 2 // 5, n_epochs], [0, lr_max, 0])[0]
#     elif lr_schedule_type == 'piecewise':
#         def lr_schedule(t):
#             if n_epochs == 0:
#                 return lr_max
#             if t / n_epochs < 0.34:#0.6:
#                 return lr_max
#             elif t / n_epochs < 0.67:#< 0.9:
#                 return lr_max / 10.
#             else:
#                 return lr_max / 100.
#     else:
#         raise ValueError('wrong lr_schedule_type')
#     return (lr_schedule_type, lr_schedule)