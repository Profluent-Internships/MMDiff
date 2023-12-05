import argparse
import data_loader
import os

TRUNK_PARAMS = ['n_extra_block', 'n_main_block', 'n_ref_block',\
                'd_msa', 'd_msa_full', 'd_pair', 'd_templ',\
                'n_head_msa', 'n_head_pair', 'n_head_templ', 'd_hidden', 'd_hidden_templ', 'p_drop']

SE3_PARAMS = ['num_layers', 'num_channels', 'num_degrees', 'n_heads', 'div', 
              'l0_in_features', 'l0_out_features', 'l1_in_features', 'l1_out_features', 'num_edge_features'
             ]

def get_args():
    parser = argparse.ArgumentParser()

    # training parameters
    train_group = parser.add_argument_group("training parameters")
    train_group.add_argument("-model_name", default=None,
            help="model name for saving")
    train_group.add_argument('-batch_size', type=int, default=1,
            help="Batch size [1]")
    train_group.add_argument('-lr', type=float, default=2.0e-4, 
            help="Learning rate [5.0e-4]")
    train_group.add_argument('-num_epochs', type=int, default=300,
            help="Number of epochs [300]")
    train_group.add_argument("-step_lr", type=int, default=300,
            help="Parameter for Step LR scheduler [300]")
    train_group.add_argument("-port", type=int, default=12319,
            help="PORT for ddp training, should be randomized [12319]")
    train_group.add_argument("-accum", type=int, default=1,
            help="Gradient accumulation when it's > 1 [1]")
    train_group.add_argument("-eval", action='store_true', default=False,
            help="Train structure only")

    # data-loading parameters
    data_group = parser.add_argument_group("data loading parameters")
    data_group.add_argument('-maxseq', type=int, default=1024,
            help="Maximum depth of subsampled MSA [1024]")
    data_group.add_argument('-maxtoken', type=int, default=2**18,
            help="Maximum depth of subsampled MSA [2**18]")
    data_group.add_argument('-maxlat', type=int, default=128,
            help="Maximum depth of subsampled MSA [128]")
    data_group.add_argument("-crop", type=int, default=260,
            help="Upper limit of crop size [260]")
    data_group.add_argument("-rescut", type=float, default=4.5,
            help="Resolution cutoff [4.5]")
    data_group.add_argument("-slice", type=str, default="DISCONT",
            help="How to make crops [CONT / DISCONT (default)]")
    data_group.add_argument("-subsmp", type=str, default="UNI",
            help="How to subsample MSAs [UNI (default) / LOG / CONST]")
    data_group.add_argument('-mintplt', type=int, default=1,
            help="Minimum number of templates to select [1]")
    data_group.add_argument('-maxtplt', type=int, default=4,
            help="maximum number of templates to select [4]")
    data_group.add_argument('-seqid', type=float, default=150.0,
            help="maximum sequence identity cutoff for template selection [150.0]")
    data_group.add_argument('-maxcycle', type=int, default=4,
            help="maximum number of recycle [4]")

    # Trunk module properties
    trunk_group = parser.add_argument_group("Trunk module parameters")
    trunk_group.add_argument('-n_extra_block', type=int, default=4,
            help="Number of iteration blocks for extra sequences [4]")
    trunk_group.add_argument('-n_main_block', type=int, default=8,
            help="Number of iteration blocks for main sequences [8]")
    trunk_group.add_argument('-n_ref_block', type=int, default=4,
            help="Number of refinement layers")
    trunk_group.add_argument('-d_msa', type=int, default=256,
            help="Number of MSA features [256]")
    trunk_group.add_argument('-d_msa_full', type=int, default=64,
            help="Number of MSA features [64]")
    trunk_group.add_argument('-d_pair', type=int, default=128,
            help="Number of pair features [128]")
    trunk_group.add_argument('-d_templ', type=int, default=64,
            help="Number of templ features [64]")
    trunk_group.add_argument('-n_head_msa', type=int, default=8,
            help="Number of attention heads for MSA2MSA [8]")
    trunk_group.add_argument('-n_head_pair', type=int, default=4,
            help="Number of attention heads for Pair2Pair [4]")
    trunk_group.add_argument('-n_head_templ', type=int, default=4,
            help="Number of attention heads for template [4]")
    trunk_group.add_argument("-d_hidden", type=int, default=32,
            help="Number of hidden features [32]")
    trunk_group.add_argument("-d_hidden_templ", type=int, default=64,
            help="Number of hidden features for templates [64]")
    trunk_group.add_argument("-p_drop", type=float, default=0.15,
            help="Dropout ratio [0.15]")

    # Structure module properties
    str_group = parser.add_argument_group("structure module parameters")
    str_group.add_argument('-num_layers', type=int, default=1,
            help="Number of equivariant layers in structure module block [1]")
    str_group.add_argument('-num_channels', type=int, default=32,
            help="Number of channels [32]")
    str_group.add_argument('-num_degrees', type=int, default=2,
            help="Number of degrees for SE(3) network [2]")
    str_group.add_argument('-l0_in_features', type=int, default=64,
            help="Number of type 0 input features [64]")
    str_group.add_argument('-l0_out_features', type=int, default=64,
            help="Number of type 0 output features [64]")
    str_group.add_argument('-l1_in_features', type=int, default=3,
            help="Number of type 1 input features [3]")
    str_group.add_argument('-l1_out_features', type=int, default=2,
            help="Number of type 1 output features [2]")
    str_group.add_argument('-num_edge_features', type=int, default=64,
            help="Number of edge features [64]")
    str_group.add_argument('-n_heads', type=int, default=4,
            help="Number of attention heads for SE3-Transformer [4]")
    str_group.add_argument("-div", type=int, default=4,
            help="Div parameter for SE3-Transformer [4]")
    str_group.add_argument('-ref_num_layers', type=int, default=2,
            help="Number of equivariant layers in structure module block [2]")
    str_group.add_argument('-ref_num_channels', type=int, default=32,
            help="Number of channels [32]")
    str_group.add_argument('-ref_l0_in_features', type=int, default=64,
            help="Number of channels [64]")
    str_group.add_argument('-ref_l0_out_features', type=int, default=64,
            help="Number of channels [64]")

    # Loss function parameters
    loss_group = parser.add_argument_group("loss parameters")
    loss_group.add_argument('-w_dist', type=float, default=1.0,
            help="Weight on distd in loss function [1.0]")
    loss_group.add_argument('-w_str', type=float, default=10.0,
            help="Weight on strd in loss function [10.0]")
    loss_group.add_argument('-w_lddt', type=float, default=0.1,
            help="Weight on predicted lddt loss [0.1]")
    loss_group.add_argument('-w_aa', type=float, default=3.0,
            help="Weight on MSA masked token prediction loss [3.0]")
    loss_group.add_argument('-w_bond', type=float, default=0.0,
            help="Weight on predicted bond loss [0.0]")
    loss_group.add_argument('-w_dih', type=float, default=0.0,
            help="Weight on pseudodihedral loss [0.0]")
    loss_group.add_argument('-w_clash', type=float, default=0.0,
            help="Weight on clash loss [0.0]")
    loss_group.add_argument('-w_hb', type=float, default=0.0,
            help="Weight on clash loss [0.0]")
    loss_group.add_argument('-w_pae', type=float, default=0.1,
            help="Weight on pae loss [0.1]")
    loss_group.add_argument('-w_bind', type=float, default=5.0,
            help="Weight on bind v no-bind prediction [5.0]")
    loss_group.add_argument('-lj_lin', type=float, default=0.75,
            help="linear inflection for lj [0.75]")

    # parse arguments
    args = parser.parse_args()

    # Setup dataloader parameters:
    loader_param = data_loader.set_data_loader_params(args)

    # make dictionary for each parameters
    trunk_param = {}
    for param in TRUNK_PARAMS:
        trunk_param[param] = getattr(args, param)
    SE3_param = {}
    for param in SE3_PARAMS:
        if hasattr(args, param):
            SE3_param[param] = getattr(args, param)

    SE3_ref_param = SE3_param.copy()

    for param in SE3_PARAMS:
        if hasattr(args, 'ref_'+param):
            SE3_ref_param[param] = getattr(args, 'ref_'+param)

    #print (SE3_param)
    #print (SE3_ref_param)
    trunk_param['SE3_param_full'] = SE3_param 
    trunk_param['SE3_param_topk'] = SE3_ref_param 

    loss_param = {}
    for param in ['w_dist', 'w_str', 'w_aa', 'w_lddt', 'w_bond', 'w_dih', 'w_clash', 'w_hb', 'w_pae', 'lj_lin']:
        loss_param[param] = getattr(args, param)

    return args, trunk_param, loader_param, loss_param
