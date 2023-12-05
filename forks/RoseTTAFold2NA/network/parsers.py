import numpy as np
import scipy
import scipy.spatial
import string
import os,re
from os.path import exists
import random
import util
import gzip
from ffindex import *
import torch
from chemical import aa2num, aa2long, NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS
import chemical

to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V',
    "DA":'a', "DC":'c', "DG":'g', "DT":'t',
    "A":'b', "C":'d', "G":'h', "U":'u',
}

def read_template_pdb(L, pdb_fn, target_chain=None, target_conf=1.0):
    # get full sequence from given PDB
    seq_full = list()
    prev_chain=''
    with open(pdb_fn) as fp:
        for line in fp:
            if line[:4] != "ATOM":
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != prev_chain:
                if len(seq_full) > 0:
                    L_s.append(len(seq_full)-offset)
                    offset = len(seq_full)
            prev_chain = line[21]
            aa = line[17:20]
            seq_full.append(aa2num[aa] if aa in aa2num.keys() else 20)

    seq_full = torch.tensor(seq_full).long()

    xyz = torch.full((L, NTOTAL, 3), np.nan).float()
    seq = torch.full((L,), 20).long()
    conf = torch.zeros(L,1).float()
    
    with open(pdb_fn) as fp:
        for line in fp:
            if line[:4] != "ATOM":
                continue
            resNo, atom, aa = int(line[22:26]), line[12:16], line[17:20]
            aa_idx = aa2num[aa] if aa in aa2num.keys() else 20
            #
            idx = resNo - 1
            for i_atm, tgtatm in enumerate(aa2long[aa_idx]):
                if tgtatm == atom:
                    xyz[idx, i_atm, :] = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    break
            seq[idx] = aa_idx
    
    mask = torch.logical_not(torch.isnan(xyz[:,:3,0])) # (L, 3)
    mask = mask.all(dim=-1)[:,None]
    conf = torch.where(mask, torch.full((L,1),target_conf), torch.zeros(L,1)).float()
    seq_1hot = torch.nn.functional.one_hot(seq, num_classes=NAATOKENS-1).float()
    t1d = torch.cat((seq_1hot, conf), -1)

    #return seq_full[None], ins[None], L_s, xyz[None], t1d[None]
    return xyz[None], t1d[None]

def parse_fasta(filename,  maxseq=10000, rna_alphabet=False, dna_alphabet=False):
    msa = []
    ins = []

    fstream = open(filename,"r")
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    table_na = {}

    for line in fstream:
        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa_i = line.translate(table)
        msa.append(msa_i)

        # sequence length
        L = len(msa[-1])

        i = np.zeros((L))
        ins.append(i)

    #msa_orig = msa.copy()

    # convert letters into numbers
    if rna_alphabet:
        alphabet = np.array(list("00000000000000000000-000000ACGUN"), dtype='|S1').view(np.uint8)
    elif dna_alphabet:
        alphabet = np.array(list("00000000000000000000-0ACGTD00000"), dtype='|S1').view(np.uint8)
    else:
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-X0000000000"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)

    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # also accept 'T' in rna_alphabet
    if rna_alphabet:
        msa[msa == ord("T")] = 30

    # fail on any illegal characters
    assert (np.all(msa<=31))

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins

def parse_mixed_fasta(filename,  maxseq=10000):
    msa1,msa2 = [],[]

    fstream = open(filename,"r")
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    unpaired_r, unpaired_p = 0, 0

    for line in fstream:
        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa_i = line.translate(table)
        msa_i = msa_i.replace('B','D') # hacky...

        msas_i = msa_i.split('/')

        if (len(msas_i)==1):
            msas_i = [msas_i[0][:len(msa1[0])], msas_i[0][len(msa1[0]):]]

        if (len(msa1)==0 or (
            len(msas_i[0])==len(msa1[0]) and len(msas_i[1])==len(msa2[0])
        )):
            # skip if we've already found half of our limit in unpaired protein seqs
            if sum([1 for x in msas_i[1] if x != '-']) == 0:
                unpaired_p += 1
                if unpaired_p > maxseq // 2:
                    continue

            # skip if we've already found half of our limit in unpaired rna seqs
            if sum([1 for x in msas_i[0] if x != '-']) == 0:
                unpaired_r += 1
                if unpaired_r > maxseq // 2:
                    continue

            msa1.append(msas_i[0])
            msa2.append(msas_i[1])
        else:
            print ("Len error",filename, len(msas_i[0]),len(msa1[0]),len(msas_i[1]),len(msas_i[1]))

        if (len(msa1) >= maxseq):
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-Xacgtxbdhuy"), dtype='|S1').view(np.uint8)
    msa1 = np.array([list(s) for s in msa1], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa1[msa1 == alphabet[i]] = i
    msa1[msa1>=31] = 21  # anything unknown to 'X'

    alphabet = np.array(list("00000000000000000000-000000ACGTN"), dtype='|S1').view(np.uint8)
    msa2 = np.array([list(s) for s in msa2], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa2[msa2 == alphabet[i]] = i
    msa2[msa2>=31] = 30  # anything unknown to 'N'

    Ls = [msa1.shape[1],msa2.shape[1]]
    msa = np.concatenate((msa1,msa2),axis=-1)
    ins = np.zeros(msa.shape, dtype=np.uint8)

    return msa,ins,Ls

# parse a fasta alignment IF it exists
# otherwise return single-sequence msa
def parse_fasta_if_exists(seq, filename, maxseq=10000, rmsa_alphabet=False):
    if (exists(filename)):
        return parse_fasta(filename, maxseq, rmsa_alphabet)
    else:
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-0acgtxbdhuy"), dtype='|S1').view(np.uint8) # -0 are UNK/mask
        seq = np.array([list(seq)], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            seq[seq == alphabet[i]] = i

        return (seq, np.zeros_like(seq))

def get_dna_msa_from_seq(seq):
    alphabet = np.array(list("00000000000000000000-0ACGT000000"), dtype='|S1').view(np.uint8)
    seq1 = np.array([list(seq)], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        seq1[seq == alphabet[i]] = i

    alphabet = np.array(list("00000000000000000000-0TGCA000000"), dtype='|S1').view(np.uint8)
    seq2 = np.array([list(seq)], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        seq2[seq == alphabet[i]] = i

    return (seq1, np.zeros_like(seq1),seq2, np.zeros_like(seq2))


# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename, unzip=True, maxseq=10000):
    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    if filename.split('.')[-1] == 'gz':
        fstream = gzip.open(filename, 'rt')
    else:
        fstream = open(filename, 'r')

    for line in fstream:

        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # remove insertion at the end
        if (not unzip):
            n_remove = 0
            for c in reversed(line):
                if c.islower():
                    n_remove += 1
                else:
                    break
            line = line[:-n_remove]

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)

        if (len(msa) >= maxseq):
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file
def parse_pdb(filename):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines)

#'''
def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), NTOTAL, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz,mask,np.array(idx_s)

def parse_pdb_w_seq(filename):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines_w_seq(lines)

def parse_pdb_lines_w_seq(lines):
    # indices of residues observed in the structure
    #idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    idx_s = [int(r[0]) for r in res]

    # chain lengths
    prev_chain=''
    seq = list()
    L_s = list()
    offset = 0

    for line in lines:
        if line[:4] != "ATOM":
            continue
        if line[12:16].strip() != "CA" and line[12:16].strip() != "P":
            continue
        if line[21] != prev_chain:
            if len(seq) > 0:
                L_s.append(len(seq)-offset)
                offset = len(seq)
        prev_chain = line[21]
        aa = line[17:20]
        seq.append(chemical.aa2num[aa] if aa in chemical.aa2num.keys() else 20)
    L_s.append(len(seq) - offset)

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), NTOTAL, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    if (len(L_s)==0):
        msa = np.array(seq)[None,...]
    else:
        msa = np.stack((seq,seq,seq), axis=0)
        msa[1,L_s[0]:] = 20
        msa[2,:L_s[0]] = 20

    msa = np.array(seq)[None,...]

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return L_s, xyz, mask, np.array(idx_s), np.array(seq), np.array(msa)


def parse_templates(item, params):

    # init FFindexDB of templates
    ### and extract template IDs
    ### present in the DB
    ffdb = FFindexDB(read_index(params['FFDB']+'_pdb.ffindex'),
                     read_data(params['FFDB']+'_pdb.ffdata'))
    #ffids = set([i.name for i in ffdb.index])

    # process tabulated hhsearch output to get
    # matched positions and positional scores
    infile = params['DIR']+'/hhr/'+item[-2:]+'/'+item+'.atab'
    hits = []
    for l in open(infile, "r").readlines():
        if l[0]=='>':
            key = l[1:].split()[0]
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]
            hits[-1][1].append([int(hi[0]),int(hi[1])])
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # get per-hit statistics from an .hhr file
    # (!!! assume that .hhr and .atab have the same hits !!!)
    # [Probab, E-value, Score, Aligned_cols, 
    # Identities, Similarity, Sum_probs, Template_Neff]
    lines = open(infile[:-4]+'hhr', "r").readlines()
    pos = [i+1 for i,l in enumerate(lines) if l[0]=='>']
    for i,posi in enumerate(pos):
        hits[i].append([float(s) for s in re.sub('[=%]',' ',lines[posi]).split()[1::2]])
        
    # parse templates from FFDB
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index)
        if entry == None:
            continue
        data = read_entry_lines(entry, ffdb.data)
        hi += list(parse_pdb_lines(data))

    # process hits
    counter = 0
    xyz,qmap,mask,f0d,f1d,ids = [],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        
        qi,ti = np.array(data[1]).T
        _,sel1,sel2 = np.intersect1d(ti, data[6], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])
        f0d.append(data[3])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[4][sel2])
        mask.append(data[5][sel2])
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))
        counter += 1

    xyz = np.vstack(xyz).astype(np.float32)
    mask = np.vstack(mask).astype(np.bool)
    qmap = np.vstack(qmap).astype(np.long)
    f0d = np.vstack(f0d).astype(np.float32)
    f1d = np.vstack(f1d).astype(np.float32)
    ids = ids
        
    return xyz,mask,qmap,f0d,f1d,ids

def parse_templates_raw(ffdb, hhr_fn, atab_fn, templ_to_use, max_templ=20):
    # process tabulated hhsearch output to get
    # matched positions and positional scores
    hits = []
    read_stat = False
    for l in open(atab_fn, "r").readlines():
        if l[0]=='>':
            read_stat = False
            if len(hits) == max_templ:
                break
            key = l[1:].split()[0]
            if len(templ_to_use) > 1:
                if key not in templ_to_use:
                    continue
            read_stat = True
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        elif not read_stat:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]
            hits[-1][1].append([int(hi[0]),int(hi[1])])
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # parse templates from FFDB
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index)
        if entry == None:
            print ("Failed to find %s in *_pdb.ffindex"%hi[0])
            continue
        data = read_entry_lines(entry, ffdb.data)
        hi += list(parse_pdb_lines_w_seq(data))[1:5] # (add four more items)

    # process hits
    counter = 0
    xyz,qmap,mask,f1d,ids,seq = [],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        #print ("Process %s..."%data[0])
        
        qi,ti = np.array(data[1]).T
        _,sel1,sel2 = np.intersect1d(ti, data[5], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[3][sel2])
        mask.append(data[4][sel2])
        seq.append(data[-1][sel2])
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))
        counter += 1
    
    if len(ids):
        # compile template features if any templates were found
        xyz = np.vstack(xyz).astype(np.float32)
        mask = np.vstack(mask).astype(np.float32)
        qmap = np.vstack(qmap).astype(np.int64)
        f1d = np.vstack(f1d).astype(np.float32)
        seq = np.hstack(seq).astype(np.int64)
        return torch.from_numpy(xyz), torch.from_numpy(mask), torch.from_numpy(qmap), \
            torch.from_numpy(f1d), torch.from_numpy(seq), ids
    else:
        return None, None, None, None, None, ids

def read_templates(qlen, ffdb, hhr_fn, atab_fn, templ_to_use=[], offset=0, n_templ=10, random_noise=5.0):
    xyz_t, mask, qmap, t1d, seq, ids = parse_templates_raw(ffdb, hhr_fn, atab_fn, templ_to_use, max_templ=max(n_templ, 20))
    npick = min(n_templ, len(ids))
    if npick < 1: # no templates
        xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(npick,qlen,1,1) + torch.rand(npick,qlen,1,3)*random_noise
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,qlen,1)).float()), -1)
        mask_t = torch.full((npick,qlen,NTOTAL),False) # True for valid atom, False for missing atom
        return xyz, t1d, mask_t

    sample = torch.arange(npick)
    #
    xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(npick,qlen,1,1) + torch.rand(npick,qlen,1,3)*random_noise
    mask_t = torch.full((npick,qlen,NTOTAL),False) # True for valid atom, False for missing atom
    f1d = torch.full((npick, qlen), 20).long() # gap token
    f1d_val = torch.zeros((npick, qlen, 1)).float()
    #
    for i, nt in enumerate(sample):
        sel = torch.where(qmap[:,1] == nt)[0]
        pos = qmap[sel, 0] + offset
        xyz[i, pos] = xyz_t[sel]
        mask_t[i,pos] = mask[sel].bool()
        f1d[i, pos] = seq[sel]
        f1d_val[i,pos] = t1d[sel, 2].unsqueeze(-1)
        #
        xyz[i] = util.center_and_realign_missing(xyz[i], mask_t[i])
    
    f1d = torch.nn.functional.one_hot(f1d, num_classes=NAATOKENS-1).float()
    f1d = torch.cat((f1d, f1d_val), dim=-1)
    
    return xyz, f1d, mask_t
