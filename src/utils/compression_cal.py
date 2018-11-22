"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/compression_cal.py
 - Contain source code for calculation compression rate and computation reduction rate.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""


def parameters_num(args):
    """
    Calculate parameters number in uncompressed and compressed model.
    :param args: arguments in compressed model
    """

    cfgs_VGG16 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512)]

    cfgs_VGG19 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512), (512, 512)]

    cfgs_MobileNet = [(3, 32), (32, 64), (64, 128, 2),
                      (128, 128), (128, 256, 2),
                      (256, 256), (256, 512, 2),
                      (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 1024, 2),
                      (1024, 1024, 2)]

    if args.model == 'MobileNet':
        cfgs = cfgs_MobileNet
    elif args.model == 'VGG16':
        cfgs = cfgs_VGG16
    elif args.model == 'VGG19':
        cfgs = cfgs_VGG19
    else:
        pass

    alpha = args.alpha
    rank = args.rank

    # uncompressed
    standard_param_num = 0
    # conv
    if args.model == 'MobileNet':
        standard_param_num += 3 * 32 * 3 * 3
    # layers
    for cfg in cfgs:
        standard_param_num += cfg[0] * cfg[1] * 3 * 3

    # compressed
    param_num = 0
    if args.convolution == 'StandardConv':
        param_num = standard_param_num
    elif args.convolution == 'MobileConv':
        # conv
        if args.model == 'MobileNet':
            param_num += 3 * 32 * 3 * 3
        # layers
        for cfg in cfgs:
            param_num += cfg[0]* 3 * 3
            param_num += cfg[0] * cfg[1]

    elif args.convolution == 'FALCON':
        # conv
        if args.model == 'MobileNet':
            param_num += 3 * 32 * 3 * 3
        # layers
        for cfg in cfgs:
            param_num += cfg[0] * cfg[1]
            param_num += cfg[1] * 3 * 3

    elif args.convolution == 'RankMobileConv':
        # conv
        if args.model == 'MobileNet':
            param_num += 3 * int(alpha * 32) * 3 * 3
        # layers
        for cfg in cfgs:
            cfg_0 = int(alpha * cfg[0])
            cfg_1 = int(alpha * cfg[1])
            param_num += cfg_0 * 3 * 3
            param_num += cfg_0 * cfg_1

        # rank
        param_num *= rank

    elif args.convolution == 'RankFALCON':
        # conv
        if args.model == 'MobileNet':
            param_num += 3 * int(alpha * 32) * 3 * 3
        # layers
        for cfg in cfgs:
            cfg_0 = int(alpha * cfg[0])
            cfg_1 = int(alpha * cfg[1])
            param_num += cfg_0 * cfg_1
            param_num += cfg_1 * 3 * 3

        # rank
        param_num *= rank

    return standard_param_num, param_num


def computation_num(args):
    """
    Calculate computation amount in uncompressed and compressed model.
    :param args: arguments in compressed model
    """

    cfgs_VGG16 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512,2)]
    maps_VGG16 = [(32, 32), (32, 16),
                  (16, 16), (16, 8),
                  (8, 8), (8, 8), (8, 4),
                  (4, 4), (4, 4), (4, 2),
                  (2, 2), (2, 2), (2, 1)]

    cfgs_VGG19 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512), (512, 512, 2)]
    maps_VGG19 = [(32, 32), (32, 16),
                  (16, 16), (16, 8),
                  (8, 8), (8, 8), (8, 4),
                  (4, 4), (4, 4), (4, 2),
                  (2, 2), (2, 2), (2, 1)]

    cfgs_MobileNet = [(3, 32), (32, 64), (64, 128, 2),
                      (128, 128), (128, 256, 2),
                      (256, 256), (256, 512, 2),
                      (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 1024, 2),
                      (1024, 1024, 2)]
    maps_MobileNet = [(32, 32), (32, 16),
                      (16, 16), (16, 8),
                      (8, 8), (8, 4),
                      (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 2),
                      (2, 1)]

    if args.model == 'MobileNet':
        cfgs = cfgs_MobileNet
        maps = maps_MobileNet
    elif args.model == 'VGG16':
        cfgs = cfgs_VGG16
        maps = maps_VGG16
    elif args.model == 'VGG19':
        cfgs = cfgs_VGG19
        maps = maps_VGG19
    else:
        pass

    alpha = args.alpha
    rank = args.rank

    standard_computation = 0
    if args.model == 'MobileNet':
        standard_computation += 3 * 3 * 3 * 32 * 32 * 32
    # layers
    for (cfg, map) in zip(cfgs, maps):
        standard_computation += cfg[0] * cfg[1] * 3 * 3 * map[1] * map[1]

    # compressed
    computation = 0
    if args.convolution == 'StandardConv':
        computation = standard_computation
    elif args.convolution == 'MobileConv':
        if args.model == 'MobileNet':
            computation += 3 * 3 * 3 * 32 * 32 * 32
        # layers
        for (cfg, map) in zip(cfgs, maps):
            computation += cfg[0] * 3 * 3 * map[1] * map[1]
            computation += cfg[0] * cfg[1] * map[1] * map[1]

    elif args.convolution == 'FALCON':
        if args.model == 'MobileNet':
            computation += 3 * 3 * 3 * 32 * 32 * 32
        # layers
        for (cfg, map) in zip(cfgs, maps):
            computation += cfg[0] * cfg[1] * map[0] * map[0]
            computation += cfg[1] * 3 * 3 * map[1] * map[1]

    elif args.convolution == 'RankMobileConv':
        if args.model == 'MobileNet':
            computation += 3 * 3 * 3 * 32 * 32 * 32
        # layers
        for (cfg, map) in zip(cfgs, maps):
            cfg_0 = int(alpha * cfg[0])
            cfg_1 = int(alpha * cfg[1])
            computation += cfg_0 * 3 * 3 * map[1] * map[1]
            computation += cfg_0 * cfg_1 * map[1] * map[1]
        # rank
        computation *= rank

    elif args.convolution == 'RankFALCON':
        if args.model == 'MobileNet':
            computation += 3 * 3 * 3 * 32 * 32 * 32
        # layers
        for (cfg, map) in zip(cfgs, maps):
            cfg_0 = int(alpha * cfg[0])
            cfg_1 = int(alpha * cfg[1])
            computation += cfg_0 * cfg_1 * map[0] * map[0]
            computation += cfg_1 * 3 * 3 * map[1] * map[1]
        # rank
        computation *= rank

    return standard_computation, computation


def cr_crr(args):
    """
    Calculate compression rate and computation reduction rate.
    :param args: arguments in compressed model
    """
    standard_param_num, param_num = parameters_num(args)
    standard_computation, computation = computation_num(args)
    print('CR = %f & CRR = %f'
          % ((float(standard_param_num)/float(param_num)),
             (float(standard_computation)/float(computation))))
