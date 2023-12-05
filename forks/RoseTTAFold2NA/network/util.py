import sys

import numpy as np
import torch

import scipy.sparse
from scipy.spatial.transform import Rotation

from chemical import *
from scoring import *

def random_rot_trans(xyz, random_noise=20.0):
    # xyz: (N, L, 27, 3)
    N, L = xyz.shape[:2]

    # pick random rotation axis
    R_mat = torch.tensor(Rotation.random(N).as_matrix(), dtype=xyz.dtype).to(xyz.device)
    xyz = torch.einsum('nij,nlaj->nlai', R_mat, xyz) + torch.rand(N,1,1,3, device=xyz.device)*random_noise
    return xyz

def center_and_realign_missing(xyz, mask_t):
    # xyz: (L, 27, 3)
    # mask_t: (L, 27)
    L = xyz.shape[0]
    
    mask = mask_t[:,:3].all(dim=-1) # True for valid atom (L)
    
    # center c.o.m at the origin
    center_CA = (mask[...,None]*xyz[:,1]).sum(dim=0) / (mask[...,None].sum(dim=0) + 1e-5) # (3)
    xyz = torch.where(mask.view(L,1,1), xyz - center_CA.view(1, 1, 3), xyz)
    
    # move missing residues to the closest valid residues
    exist_in_xyz = torch.where(mask)[0] # L_sub
    seqmap = (torch.arange(L, device=xyz.device)[:,None] - exist_in_xyz[None,:]).abs() # (L, Lsub)
    seqmap = torch.argmin(seqmap, dim=-1) # L
    idx = torch.gather(exist_in_xyz, 0, seqmap)
    offset_CA = torch.gather(xyz[:,1], 0, idx.reshape(L,1).expand(-1,3))
    xyz = torch.where(mask.view(L,1,1), xyz, xyz + offset_CA.reshape(L,1,3))

    return xyz

def th_ang_v(ab,bc,eps:float=1e-8):
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih_v(ab,bc,cd):
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih(a,b,c,d):
    return th_dih_v(a-b,b-c,c-d)

# build a frame from 3 points
#fd  -  more complicated version splits angle deviations between CA-N and CA-C (giving more accurate CB position)
#fd  -  makes no assumptions about input dims (other than last 1 is xyz)
def rigid_from_3_points(N, Ca, C, is_na=None, eps=1e-8):
    dims = N.shape[:-1]

    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('...li, ...li -> ...l', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
    v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
    cosref = torch.sum(e1*v2, dim=-1)
    costgt = torch.full(dims, -0.3616, device=N.device)
    if is_na is not None:
        costgt[is_na] = -0.4929

    cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
    cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
    sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
    Rp = torch.eye(3, device=N.device).repeat(*dims,1,1)
    Rp[...,0,0] = cosdel
    Rp[...,0,1] = -sindel
    Rp[...,1,0] = sindel
    Rp[...,1,1] = cosdel

    R = torch.einsum('...ij,...jk->...ik', R,Rp)

    return R, Ca

# note: needs consistency with chemical.py
def is_nucleic(seq):
    return (seq>=22)

# note: needs consistency with chemical.py
def dna_reverse_complement(seq):
    rseq = 47 - torch.flip(seq,(-1,))
    return rseq

def idealize_reference_frame(seq, xyz_in):
    xyz = xyz_in.clone()

    namask = is_nucleic(seq)
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], namask)

    protmask = ~namask

    Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
    Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)

    OP1ideal = torch.tensor([-0.7319, 1.2920, 0.000], device=xyz_in.device)
    OP2ideal = torch.tensor([1.4855, 0.000, 0.000], device=xyz_in.device)

    pmask_bs,pmask_rs = protmask.nonzero(as_tuple=True)
    nmask_bs,nmask_rs = namask.nonzero(as_tuple=True)
    xyz[pmask_bs,pmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], Nideal) + Ts[pmask_bs,pmask_rs]
    xyz[pmask_bs,pmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], Cideal) + Ts[pmask_bs,pmask_rs]
    xyz[nmask_bs,nmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], OP1ideal) + Ts[nmask_bs,nmask_rs]
    xyz[nmask_bs,nmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], OP2ideal) + Ts[nmask_bs,nmask_rs]

    return xyz

def get_frames(xyz_in, xyz_mask, seq, frame_indices):
    B,L = xyz_in.shape[:2]
    frames = frame_indices[seq]

    frame_mask = frames[...,0] != frames[...,1]
    frame_mask *= torch.all(
        torch.gather(xyz_mask,2,frames.reshape(B,L,-1)).reshape(B,L,-1,3),
        axis=-1)

    return frames, frame_mask

def generate_Cbeta(N,Ca,C):
    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    #Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # fd: below matches sidechain generator (=Rosetta params)
    Cb = -0.57910144*a + 0.5689693*b - 0.5441217*c + Ca

    return Cb

def get_tips(xyz, seq):
    B,L = xyz.shape[:2]

    xyz_tips = torch.gather(xyz, 2, tip_indices.to(xyz.device)[seq][:,:,None,None].expand(-1,-1,-1,3)).reshape(B, L, 3)
    if torch.isnan(xyz_tips).any(): # replace NaN tip atom with virtual Cb atom
        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

        xyz_tips = torch.where(torch.isnan(xyz_tips), Cb, xyz_tips)
    return xyz_tips


# writepdb
def writepdb(filename, atoms, seq, Ls, idx_pdb=None, bfacts=None):
    f = open(filename,"w")
    ctr = 1
    scpu = seq.cpu().squeeze(0)
    atomscpu = atoms.cpu().squeeze(0)

    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp( bfacts.cpu(), 0, 100)
    chn_idx, res_idx = 0, 0
    for i,s in enumerate(scpu):
        natoms = atomscpu.shape[-2]
        if (natoms!=NHEAVY and natoms!=NTOTAL):
            print ('bad size!', natoms, NHEAVY, NTOTAL, atoms.shape)
            assert(False)

        atms = aa2long[s]

        # his protonation state hack
        if (s==8 and torch.linalg.norm( atomscpu[i,9,:]-atomscpu[i,5,:] ) < 1.7):
            atms = (
                " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                " HD1",  None,  None,  None,  None,  None,  None) # his_d

        for j,atm_j in enumerate(atms):
            if (j<natoms and atm_j is not None and not torch.isnan(atomscpu[i,j,:]).any()):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm_j, num2aa[s], 
                    PDB_CHAIN_IDS[chn_idx], res_idx+1, atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                    1.0, Bfacts[i] ) )
                ctr += 1

        res_idx += 1
        if (chn_idx < len(Ls) and res_idx == Ls[chn_idx]):
            chn_idx += 1
            res_idx = 0

######
######

# process ideal frames
def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn)
    Zn =  Z / torch.linalg.norm(Z)
    return torch.stack((Xn,Yn,Zn), dim=-1)


# resolve tip atom indices
tip_indices = torch.full((NAATOKENS,), 0)
for i in range(NAATOKENS):
    tip_atm = aa2tip[i]
    atm_long = aa2long[i]
    tip_indices[i] = atm_long.index(tip_atm)

# resolve torsion indices
#  a negative index indicates the previous residue
# order:
#    omega/phi/psi: 0-2
#    chi_1-4(prot): 3-6
#    cb/cg bend: 7-9
#    eps(p)/zeta(p): 10-11
#    alpha/beta/gamma/delta: 12-15
#    nu2/nu1/nu0: 16-18
#    chi_1(na): 19
torsion_indices = torch.full((NAATOKENS,NTOTALDOFS,4),0)
torsion_can_flip = torch.full((NAATOKENS,NTOTALDOFS),False,dtype=torch.bool)
for i in range(NPROTAAS):
    i_l, i_a = aa2long[i], aa2longalt[i]

    # protein omega/phi/psi
    torsion_indices[i,0,:] = torch.tensor([-1,-2,0,1]) # omega
    torsion_indices[i,1,:] = torch.tensor([-2,0,1,2]) # phi
    torsion_indices[i,2,:] = torch.tensor([0,1,2,3]) # psi (+pi)

    # protein chis
    for j in range(4):
        if torsions[i][j] is None:
            continue
        for k in range(4):
            a = torsions[i][j][k]
            torsion_indices[i,3+j,k] = i_l.index(a)
            if (i_l.index(a) != i_a.index(a)):
                torsion_can_flip[i,3+j] = True ##bb tors never flip

    # CB/CG angles (only masking uses these indices)
    torsion_indices[i,7,:] = torch.tensor([0,2,1,4]) # CB ang1
    torsion_indices[i,8,:] = torch.tensor([0,2,1,4]) # CB ang2
    torsion_indices[i,9,:] = torch.tensor([0,2,4,5]) # CG ang (arg 1 ignored)

# HIS is a special case for flip
torsion_can_flip[8,4]=False

for i in range(NPROTAAS,NAATOKENS):
    # NA BB tors
    torsion_indices[i,10,:] = torch.tensor([-5,-7,-8,1])  # epsilon_prev
    torsion_indices[i,11,:] = torch.tensor([-7,-8,1,3])   # zeta_prev
    torsion_indices[i,12,:] = torch.tensor([0,1,3,4])     # alpha (+2pi/3)
    torsion_indices[i,13,:] = torch.tensor([1,3,4,5])     # beta
    torsion_indices[i,14,:] = torch.tensor([3,4,5,7])     # gamma
    torsion_indices[i,15,:] = torch.tensor([4,5,7,8])     # delta

    if (i<NPROTAAS+5):
        # is DNA
        torsion_indices[i,16,:] = torch.tensor([4,5,6,10])     # nu2
        torsion_indices[i,17,:] = torch.tensor([5,6,10,9])     # nu1
        torsion_indices[i,18,:] = torch.tensor([6,10,9,7])     # nu0
    else:   
        # is RNA (fd: my fault since I flipped C1'/C2' order for DNA and RNA)
        torsion_indices[i,16,:] = torch.tensor([4,5,6,9])     # nu2
        torsion_indices[i,17,:] = torch.tensor([5,6,9,10])     # nu1
        torsion_indices[i,18,:] = torch.tensor([6,9,10,7])     # nu0

    # NA chi
    if torsions[i][0] is not None:
        i_l = aa2long[i]
        for k in range(4):
            a = torsions[i][0][k]
            torsion_indices[i,19,k] = i_l.index(a) # chi
        # no NA torsion flips

# build the mapping from atoms in the full rep (Nx27) to the "alternate" rep
allatom_mask = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.bool)
long2alt = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)
for i in range(NAATOKENS):
    i_l, i_lalt = aa2long[i],  aa2longalt[i]
    for j,a in enumerate(i_l):
        if (a is None):
            long2alt[i,j] = j
        else:
            long2alt[i,j] = i_lalt.index(a)
            allatom_mask[i,j] = True

# bond graph traversal
num_bonds = torch.zeros((NAATOKENS,NTOTAL,NTOTAL), dtype=torch.long)
for i in range(NAATOKENS):
    num_bonds_i = np.zeros((NTOTAL,NTOTAL))
    for (bnamei,bnamej) in aabonds[i]:
        bi,bj = aa2long[i].index(bnamei),aa2long[i].index(bnamej)
        num_bonds_i[bi,bj] = 1
    num_bonds_i = scipy.sparse.csgraph.shortest_path (num_bonds_i,directed=False)
    num_bonds_i[num_bonds_i>=4] = 4
    num_bonds[i,...] = torch.tensor(num_bonds_i)


# atom type indices
idx2aatype = []
for x in aa2type:
    for y in x:
        if y and y not in idx2aatype:
            idx2aatype.append(y)
aatype2idx = {x:i for i,x in enumerate(idx2aatype)}

# element indices
idx2elt = []
for x in aa2elt:
    for y in x:
        if y and y not in idx2elt:
            idx2elt.append(y)
elt2idx = {x:i for i,x in enumerate(idx2elt)}

# LJ/LK scoring parameters
atom_type_index = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)
element_index = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)

ljlk_parameters = torch.zeros((NAATOKENS,NTOTAL,5), dtype=torch.float)
lj_correction_parameters = torch.zeros((NAATOKENS,NTOTAL,4), dtype=bool) # donor/acceptor/hpol/disulf
for i in range(NAATOKENS):
    for j,a in enumerate(aa2type[i]):
        if (a is not None):
            atom_type_index[i,j] = aatype2idx[a]
            ljlk_parameters[i,j,:] = torch.tensor( type2ljlk[a] )
            lj_correction_parameters[i,j,0] = (type2hb[a]==HbAtom.DO)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,1] = (type2hb[a]==HbAtom.AC)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,2] = (type2hb[a]==HbAtom.HP)
            lj_correction_parameters[i,j,3] = (a=="SH1" or a=="HS")
    for j,a in enumerate(aa2elt[i]):
        if (a is not None):
            element_index[i,j] = elt2idx[a]

# hbond scoring parameters
def donorHs(D,bonds,atoms):
    dHs = []
    for (i,j) in bonds:
        if (i==D):
            idx_j = atoms.index(j)
            if (idx_j>=NHEAVY):  # if atom j is a hydrogen
                dHs.append(idx_j)
        if (j==D):
            idx_i = atoms.index(i)
            if (idx_i>=NHEAVY):  # if atom j is a hydrogen
                dHs.append(idx_i)
    assert (len(dHs)>0)
    return dHs

def acceptorBB0(A,hyb,bonds,atoms):
    if (hyb == HbHybType.SP2):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<NHEAVY):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<NHEAVY):
                    break
        for (i,j) in bonds:
            if (i==atoms[B]):
                B0 = atoms.index(j)
                if (B0<NHEAVY):
                    break
            if (j==atoms[B]):
                B0 = atoms.index(i)
                if (B0<NHEAVY):
                    break
    elif (hyb == HbHybType.SP3 or hyb == HbHybType.RING):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<NHEAVY):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<NHEAVY):
                    break
        for (i,j) in bonds:
            if (i==A and j!=atoms[B]):
                B0 = atoms.index(j)
                break
            if (j==A and i!=atoms[B]):
                B0 = atoms.index(i)
                break

    return B,B0


hbtypes = torch.full((NAATOKENS,NTOTAL,3),-1, dtype=torch.long) # (donortype, acceptortype, acchybtype)
hbbaseatoms = torch.full((NAATOKENS,NTOTAL,2),-1, dtype=torch.long) # (B,B0) for acc; (D,-1) for don
hbpolys = torch.zeros((HbDonType.NTYPES,HbAccType.NTYPES,3,15)) # weight,xmin,xmax,ymin,ymax,c9,...,c0

for i in range(NAATOKENS):
    for j,a in enumerate(aa2type[i]):
        if (a in type2dontype):
            j_hs = donorHs(aa2long[i][j],aabonds[i],aa2long[i])
            for j_h in j_hs:
                hbtypes[i,j_h,0] = type2dontype[a]
                hbbaseatoms[i,j_h,0] = j
        if (a in type2acctype):
            j_b, j_b0 = acceptorBB0(aa2long[i][j],type2hybtype[a],aabonds[i],aa2long[i])
            hbtypes[i,j,1] = type2acctype[a]
            hbtypes[i,j,2] = type2hybtype[a]
            hbbaseatoms[i,j,0] = j_b
            hbbaseatoms[i,j,1] = j_b0

for i in range(HbDonType.NTYPES):
    for j in range(HbAccType.NTYPES):
        weight = dontype2wt[i]*acctype2wt[j]

        pdist,pbah,pahd = hbtypepair2poly[(i,j)]
        xrange,yrange,coeffs = hbpolytype2coeffs[pdist]
        hbpolys[i,j,0,0] = weight
        hbpolys[i,j,0,1:3] = torch.tensor(xrange)
        hbpolys[i,j,0,3:5] = torch.tensor(yrange)
        hbpolys[i,j,0,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pahd]
        hbpolys[i,j,1,0] = weight
        hbpolys[i,j,1,1:3] = torch.tensor(xrange)
        hbpolys[i,j,1,3:5] = torch.tensor(yrange)
        hbpolys[i,j,1,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pbah]
        hbpolys[i,j,2,0] = weight
        hbpolys[i,j,2,1:3] = torch.tensor(xrange)
        hbpolys[i,j,2,3:5] = torch.tensor(yrange)
        hbpolys[i,j,2,5:] = torch.tensor(coeffs)

# kinematic parameters
base_indices = torch.full((NAATOKENS,NTOTAL),0, dtype=torch.long) # base frame that builds each atom
xyzs_in_base_frame = torch.ones((NAATOKENS,NTOTAL,4)) # coords of each atom in the base frame
RTs_by_torsion = torch.eye(4).repeat(NAATOKENS,NTOTALTORS,1,1) # torsion frames
reference_angles = torch.ones((NAATOKENS,NPROTANGS,2)) # reference values for bendable angles

## PROTEIN
for i in range(NPROTAAS):
    i_l = aa2long[i]
    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # omega frame
    RTs_by_torsion[i,0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,0,:3,3] = torch.zeros(3)

    # phi frame
    RTs_by_torsion[i,1,:3,:3] = make_frame(
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3],
        torch.tensor([1.,0.,0.])
    )
    RTs_by_torsion[i,1,:3,3] = xyzs_in_base_frame[i,0,:3]

    # psi frame
    RTs_by_torsion[i,2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,2,:3] - xyzs_in_base_frame[i,1,:3],
        xyzs_in_base_frame[i,1,:3] - xyzs_in_base_frame[i,0,:3]
    )
    RTs_by_torsion[i,2,:3,3] = xyzs_in_base_frame[i,2,:3]

    # chi1 frame
    if torsions[i][0] is not None:
        a0,a1,a2 = torsion_indices[i,3,0:3]
        RTs_by_torsion[i,3,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
            xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3],
        )
        RTs_by_torsion[i,3,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # chi2/3/4 frame
    for j in range(1,4):
        if torsions[i][j] is not None:
            a2 = torsion_indices[i,3+j,2]
            if ((i==18 and j==2) or (i==8 and j==2)):  # TYR CZ-OH & HIS CE1-HE1 a special case
                a0,a1 = torsion_indices[i,3+j,0:2]
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
                    xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3] )
            else:
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3],
                    torch.tensor([-1.,0.,0.]), )
            RTs_by_torsion[i,3+j,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # CB/CG angles
    NCr = 0.5*(xyzs_in_base_frame[i,0,:3]+xyzs_in_base_frame[i,2,:3])
    CAr = xyzs_in_base_frame[i,1,:3]
    CBr = xyzs_in_base_frame[i,4,:3]
    CGr = xyzs_in_base_frame[i,5,:3]
    reference_angles[i,0,:]=th_ang_v(CBr-CAr,NCr-CAr)
    NCp = xyzs_in_base_frame[i,2,:3]-xyzs_in_base_frame[i,0,:3]
    NCpp = NCp - torch.dot(NCp,NCr)/ torch.dot(NCr,NCr) * NCr
    reference_angles[i,1,:]=th_ang_v(CBr-CAr,NCpp)
    reference_angles[i,2,:]=th_ang_v(CGr,torch.tensor([-1.,0.,0.]))

## NUCLEIC ACIDS
for i in range(NPROTAAS, NAATOKENS):
    i_l = aa2long[i]

    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # epsilon(p)/zeta(p) - like omega in protein, not used to build atoms
    #                    - keep as identity
    RTs_by_torsion[i,NPROTTORS+0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,NPROTTORS+0,:3,3] = torch.zeros(3)
    RTs_by_torsion[i,NPROTTORS+1,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,NPROTTORS+1,:3,3] = torch.zeros(3)

    # alpha
    RTs_by_torsion[i,NPROTTORS+2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,3,:3] - xyzs_in_base_frame[i,1,:3], # P->O5'
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3]  # P<-OP1
    )
    RTs_by_torsion[i,NPROTTORS+2,:3,3] = xyzs_in_base_frame[i,3,:3] # O5'

    # beta
    RTs_by_torsion[i,NPROTTORS+3,:3,:3] = make_frame(
        xyzs_in_base_frame[i,4,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+3,:3,3] = xyzs_in_base_frame[i,4,:3] # C5'

    # gamma
    RTs_by_torsion[i,NPROTTORS+4,:3,:3] = make_frame(
        xyzs_in_base_frame[i,5,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+4,:3,3] = xyzs_in_base_frame[i,5,:3] # C4'

    # delta
    RTs_by_torsion[i,NPROTTORS+5,:3,:3] = make_frame(
        xyzs_in_base_frame[i,7,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+5,:3,3] = xyzs_in_base_frame[i,7,:3] # C3'

    # nu2
    RTs_by_torsion[i,NPROTTORS+6,:3,:3] = make_frame(
        xyzs_in_base_frame[i,6,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+6,:3,3] = xyzs_in_base_frame[i,6,:3] # O4'

    # nu1
    if i<NPROTAAS+5:
        # is DNA
        C1idx,C2idx = 10,9
    else:
        # is RNA
        C1idx,C2idx = 9,10

    RTs_by_torsion[i,NPROTTORS+7,:3,:3] = make_frame(
        xyzs_in_base_frame[i,C1idx,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+7,:3,3] = xyzs_in_base_frame[i,C1idx,:3] # C1'

    # nu0
    RTs_by_torsion[i,NPROTTORS+8,:3,:3] = make_frame(
        xyzs_in_base_frame[i,C2idx,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+8,:3,3] = xyzs_in_base_frame[i,C2idx,:3] # C2'

    # NA chi
    if torsions[i][0] is not None:
        a2 = torsion_indices[i,19,2]
        RTs_by_torsion[i,NPROTTORS+9,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,NPROTTORS+9,:3,3] = xyzs_in_base_frame[i,a2,:3]

# general FAPE parameters
frame_indices = torch.full((NAATOKENS,NFRAMES,3),0, dtype=torch.long)
for i in range(NAATOKENS):
    i_l = aa2long[i]
    for j,x in enumerate(frames[i]):
        if x is not None:
            frame_indices[i,j,0] = i_l.index(x[0])
            frame_indices[i,j,1] = i_l.index(x[1])
            frame_indices[i,j,2] = i_l.index(x[2])


