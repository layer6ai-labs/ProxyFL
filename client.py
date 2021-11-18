import torch
import torch.nn as nn

from model import LeNet5, MLP, CNN1, CNN2
from opacus import PrivacyEngine


def init_model(model_type, in_channel, n_class):

    if model_type == 'LeNet5':
        model = LeNet5(in_channel, n_class)
    elif model_type == 'MLP':
        model = MLP(n_class)
    elif model_type == 'CNN1':
        model = CNN1(in_channel, n_class)
    elif model_type == 'CNN2':
        model = CNN2(in_channel, n_class)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    return model


def init_optimizer(model, args):

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=1e-4)
    else:
        raise ValueError("Unknown optimizer")

    return optimizer


def init_dp_optimizer(model, data_size, args):
    opt = init_optimizer(model, args)
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=args.batch_size / data_size,
        alphas=orders,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.l2_norm_clip,
    )
    # print(f"Using DP-SGD with sigma={args.noise_multiplier} and clipping norm max={args.l2_norm_clip}")
    privacy_engine.attach(opt)
    return opt
            
            
class Client(nn.Module):

    def __init__(self, data, args):

        super(Client, self).__init__()

        self.private_data = data

        self.private_model = init_model(args.private_model_type, args.in_channel, args.n_class).to(args.device)
        self.proxy_model = init_model(args.proxy_model_type, args.in_channel, args.n_class).to(args.device)

        if args.use_private_SGD:
            # Using DP proxies, train private model without DP
            self.proxy_opt = init_dp_optimizer(self.proxy_model, self.private_data[0].shape[0], args)
        else:
            self.proxy_opt = init_optimizer(self.proxy_model, args)

        # Private model training always not DP
        self.private_opt = init_optimizer(self.private_model, args)

        self.device = args.device
        self.tot_epochs = 0
        self.privacy_budget = 0.
