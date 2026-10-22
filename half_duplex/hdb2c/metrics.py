"""Explicit hypothesis and tensor-comparison metrics; gates never auto-pass."""
import re
import unicodedata
import numpy as np


def normalize_text(s):
    return ' '.join(re.findall(r"\w+(?:['’]\w+)?",unicodedata.normalize('NFC',s).casefold()))


def words(s):
    return re.findall(r"\w+(?:['’]\w+)?",normalize_text(s))


def distance(a,b):
    old=list(range(len(b)+1))
    for i,token in enumerate(a,1):
        row=[i]
        for j,other in enumerate(b,1):
            row.append(min(old[j]+1,row[j-1]+1,old[j-1]+(token!=other)))
        old=row
    return old[-1]


def wer(reference,hypothesis):
    ref=words(reference)
    if not ref: raise ValueError('Empty manually verified transcript')
    return {'word_errors':distance(ref,words(hypothesis)), 'reference_words':len(ref),
            'wer':distance(ref,words(hypothesis))/len(ref)}


def compare_arrays(a,b,kind,atol=1e-3,min_cosine=0.999):
    """Explicit parity gate on same-order, SAME-SHAPE valid tensors."""
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape:
        return {'status':'SHAPE_MISMATCH','kind':kind,'nemo_shape':list(a.shape),
                'candle_shape':list(b.shape)}
    if a.size==0: return {'status':'NO_VALUES','kind':kind}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'status':'NONFINITE','kind':kind}
    d=a-b
    aa=a.ravel();bb=b.ravel()
    na=np.linalg.norm(aa);nb=np.linalg.norm(bb)
    cos=float(np.dot(aa,bb)/(na*nb)) if na*nb>0 else (1.0 if na==nb else 0.0)
    rmse=float(np.sqrt(np.mean(d*d)))
    maximum=float(np.max(np.abs(d)))
    passed=(maximum<=atol and cos>=min_cosine)
    return {'status':'PASS_NUMERIC_PARITY' if passed else 'NUMERIC_MISMATCH',
            'kind':kind,'shape':list(a.shape),'max_abs':maximum,'rmse':rmse,
            'mean_abs':float(np.mean(np.abs(d))), 'cosine':cos,
            'atol':atol,'min_cosine':min_cosine}
