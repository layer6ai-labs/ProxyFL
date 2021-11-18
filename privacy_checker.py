from opacus.privacy_analysis import compute_rdp, get_privacy_spent


def check_privacy(args):

    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))

    delta = 1.0 / args.n_client_data
    sample_rate = args.batch_size / args.n_client_data
    noise_multiplier = args.noise_multiplier

    steps = args.n_rounds * args.n_epochs / sample_rate

    rdps = compute_rdp(sample_rate, noise_multiplier, steps, orders)
    epsilon, alpha = get_privacy_spent(orders, rdps, delta)

    return epsilon, alpha
